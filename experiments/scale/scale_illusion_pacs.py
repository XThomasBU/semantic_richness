"""
Scale Illusion (PACS): Scale invariance vs. semantic familiarity, using PACS-style domains.

Dataset layout expected:
  pacs_subset/
    art_painting_dog/
      art_painting_0_101.png
      ...
    cartoon_dog/
    photo_dog/
    sketch_dog/
    ...

Folder names are assumed to be "<domain>_<object>", where <domain> can include underscores.
The <object> is taken as the final underscore-separated token.

Protocol (mirrors experiments/scale/scale_illusion.py):
  - Positive: image X (full size) vs image X (scaled with padding)
  - Negative (hard): image X (full size) vs image Y (scaled), where Y is from the same domain but a different object
  - Scale factors: [0.1, 0.3, 0.5, 0.9] by default
"""

import os
import sys
import argparse
import logging
import random
import re
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.infer import InferenceModel
from google.genai.errors import ServerError


SCALE_FACTORS = [0.1, 0.3, 0.5, 0.9]

# Prompt options (PACS version: "object" instead of "character")
PROMPT_COT = """Look at the following two images.

Do these two images show the same underlying object? One could be a scaled version of the other.

Answer with just "YES" or "NO" after your reasoning."""

PROMPT_DIRECT = """You are given two images, each containing a single object on a white background. Decide whether they depict the same underlying object, allowing for significant differences in size, scale, or resolution. Answer with exactly YES or NO."""

PROMPT_V2 = """Compare the two images and decide if they show the same object.
Ignore differences in scale, size, or resolution. Answer with exactly YES or NO."""

PROMPT_V3 = """Do the two images show the same visual object, even if their size or resolution differs? Answer YES or NO."""


def stitch_two_images_with_labels(
    image1_path: str,
    image2_path: str,
    output_path: str,
    label1: str = "image1",
    label2: str = "image2",
    margin: int = 12,
    label_height: int = 36,
    bg_color=(255, 255, 255),
):
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    w1, h1 = img1.size
    w2, h2 = img2.size
    h = max(h1, h2)

    canvas_w = w1 + w2 + margin * 3
    canvas_h = h + label_height + margin * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    x1, y1 = margin, margin
    x2, y2 = margin * 2 + w1, margin
    canvas.paste(img1, (x1, y1))
    canvas.paste(img2, (x2, y2))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label_y = margin + h + margin // 2

    def _center_text(x_left: int, panel_w: int, text: str):
        if font is None:
            tw, th = draw.textsize(text)
        else:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x_left + (panel_w - tw) // 2
        return tx, th

    tx1, th1 = _center_text(x1, w1, label1)
    tx2, th2 = _center_text(x2, w2, label2)
    ty = label_y + max(0, (label_height - max(th1, th2)) // 2)

    draw.text((tx1, ty), label1, fill=(0, 0, 0), font=font)
    draw.text((tx2, ty), label2, fill=(0, 0, 0), font=font)

    canvas.save(output_path, "PNG")
    return output_path


def setup_logging(log_path: str = None):
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if log_path:
        logging.basicConfig(filename=log_path, level=logging.INFO, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
    return logging.getLogger(__name__)


def safe_infer(model, infer_args, max_retries: int = 5):
    """
    Run model inference with automatic retry on 503 UNAVAILABLE errors.
    Implements exponential backoff to handle overloaded servers.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return model.infer(infer_args)
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = (2**attempt) + random.random() * 2
                print(
                    f"[WARN] Server overloaded (attempt {attempt}/{max_retries}). Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + random.random()
                print(
                    f"[WARN] Error during inference (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Model unavailable after {max_retries} retries. Aborting inference.")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_response(response_text: str) -> Dict[str, any]:
    """
    Parse model response to extract YES/NO answer.
    Preserves original response without stripping.
    """
    if not response_text:
        return {"answer": "unknown", "response_clean": ""}

    if isinstance(response_text, list):
        response_text = response_text[0] if response_text else ""

    response_clean = str(response_text)
    text_for_matching = str(response_text).lower().strip()

    yes_patterns = [r"\byes\b", r"\btrue\b", r"\bsame\b"]
    no_patterns = [r"\bno\b", r"\bfalse\b", r"\bdifferent\b"]

    for pattern in yes_patterns:
        if re.search(pattern, text_for_matching):
            return {"answer": "yes", "response_clean": response_clean}
    for pattern in no_patterns:
        if re.search(pattern, text_for_matching):
            return {"answer": "no", "response_clean": response_clean}
    return {"answer": "unknown", "response_clean": response_clean}


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(model_name).lower()).strip("_")


def _parse_domain_object(folder_name: str) -> Tuple[str, str]:
    parts = folder_name.split("_")
    if len(parts) < 2:
        return folder_name, "unknown"
    obj = parts[-1]
    domain = "_".join(parts[:-1])
    return domain, obj


def load_pacs_images(pacs_dir: str, max_images_per_subcategory: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Returns list of (domain, object, image_path).
    """
    root = Path(pacs_dir)
    if not root.exists():
        raise FileNotFoundError(f"PACS subset directory not found: {root}")

    entries: List[Tuple[str, str, str]] = []
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for subdir in subdirs:
        domain, obj = _parse_domain_object(subdir.name)
        image_files = sorted(list(subdir.glob("*.png")))
        if max_images_per_subcategory is not None:
            image_files = image_files[: int(max_images_per_subcategory)]
        for img_path in image_files:
            entries.append((domain, obj, str(img_path)))
    return entries


def resize_with_padding(image_path: str, scale_factor: float, output_path: str = None, image_size: int = 336) -> str:
    """
    Resize an image by adding white padding, making the object smaller within the same canvas.
    """
    img = Image.open(image_path).convert("RGB")

    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    new_size = max(1, int(image_size * scale_factor))
    img_small = img.resize((new_size, new_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    paste_x = (image_size - new_size) // 2
    paste_y = (image_size - new_size) // 2
    canvas.paste(img_small, (paste_x, paste_y))

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"resized_{random.randint(0, 999999)}.png")
    canvas.save(output_path, "PNG")
    return output_path


def create_pacs_sanity_check(
    pacs_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
    scale_factors: List[float] = None,
    max_images_per_subcategory: Optional[int] = None,
):
    """
    Create sanity check visualization: original vs scaled (with padding).
    """
    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    pacs_images = load_pacs_images(pacs_dir, max_images_per_subcategory=max_images_per_subcategory)
    if len(pacs_images) < 2:
        raise ValueError("Not enough PACS images for sanity check")

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 2.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        domain, obj, img_path = random.choice(pacs_images)
        scale_factor = random.choice(scale_factors)

        img1 = Image.open(img_path).convert("RGB")
        if img1.size != (image_size, image_size):
            img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

        tmp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_scaled.close()
        resize_with_padding(img_path, scale_factor, tmp_scaled.name, image_size=image_size)
        img2 = Image.open(tmp_scaled.name).convert("RGB")
        os.unlink(tmp_scaled.name)

        img1_display = img1.resize((150, 150), Image.Resampling.LANCZOS)
        img2_display = img2.resize((150, 150), Image.Resampling.LANCZOS)

        axes[i, 0].imshow(img1_display)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img2_display)
        axes[i, 1].axis("off")

        if i == 0:
            axes[i, 0].text(
                0.5,
                -0.1,
                "Original\n(Full Size)",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            axes[i, 1].text(
                0.5,
                -0.1,
                "Scaled\n(With Padding)",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        label = f"{domain}/{obj}\n(scale={scale_factor:.1f})"
        axes[i, 1].text(
            1.1,
            0.5,
            label,
            transform=axes[i, 1].transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PACS sanity check visualization saved to: {output_path}", flush=True)


def run_scale_illusion_pacs_experiment(
    pacs_dir: str,
    model_config: Dict[str, str] = {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct"},
    num_samples: int = 50,  # kept for backward compatibility; PACS runs over all images by default
    positive_ratio: float = 0.5,
    use_cot_prompt: bool = False,
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    scale_factors: List[float] = None,
    resume_csv: str = None,
    prompt_version: str = "direct",
    prompt_tag: str = None,
    max_images_per_subcategory: Optional[int] = None,
):
    logger = setup_logging(log_path)
    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    logger.info("=" * 80)
    logger.info("SCALE ILLUSION (PACS) EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"PACS dir: {pacs_dir}")
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info("Total samples: all images × all scales (no sampling) × {positive, negative}")
    logger.info(f"Positive ratio (unused; fixed 50/50): {positive_ratio}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Scale factors: {scale_factors}")
    logger.info(f"Output directory: {output_dir}")
    if max_images_per_subcategory is not None:
        logger.info(f"Max images per subcategory: {max_images_per_subcategory}")

    temp_dir = tempfile.mkdtemp(prefix="scale_illusion_pacs_")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        logger.info("Initializing model...")
        model_name = model_config.get("model_name", "unknown")
        model = InferenceModel(model_name)
        model_slug = _model_slug(model_name)
        logger.info(f"Model initialized: {model_name}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    logger.info("Loading PACS images...")
    pacs_images = load_pacs_images(pacs_dir, max_images_per_subcategory=max_images_per_subcategory)
    logger.info(f"Loaded {len(pacs_images)} PACS images")
    if len(pacs_images) < 2:
        raise ValueError("Not enough PACS images (need at least 2)")

    # Select prompt
    if use_cot_prompt:
        prompt_version = "cot"
    if prompt_version == "cot":
        prompt_template = PROMPT_COT
        resolved_tag = "prompt_cot"
    elif prompt_version == "v2":
        prompt_template = PROMPT_V2
        resolved_tag = "prompt_2"
    elif prompt_version == "v3":
        prompt_template = PROMPT_V3
        resolved_tag = "prompt_3"
    else:
        prompt_template = PROMPT_DIRECT
        resolved_tag = None
    if prompt_tag:
        resolved_tag = prompt_tag

    exp_dir_name = f"scale_illusion_pacs_{model_slug}" if model_name == "gemini-2.5-pro" else "scale_illusion_pacs"
    if resolved_tag:
        exp_dir_name = f"{exp_dir_name}_{resolved_tag}"
    exp_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")

    # Sanity check
    logger.info("Creating PACS sanity check visualization...")
    sanity_path = os.path.join(sanity_dir, f"{model_slug}_scale_illusion_pacs_sanity_check.png")
    try:
        create_pacs_sanity_check(
            pacs_dir=pacs_dir,
            output_path=sanity_path,
            num_examples=6,
            image_size=image_size,
            scale_factors=scale_factors,
            max_images_per_subcategory=max_images_per_subcategory,
        )
    except Exception as e:
        logger.warning(f"Failed to create sanity check: {e}")

    is_llava = model_name in ("llava", "llava-1.5-7b") or str(model_name).startswith("llava-hf/")
    if is_llava:
        try:
            domain, obj, img_path = random.choice(pacs_images)
            tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp1.close()
            tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp2.close()
            img1 = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
            img1.save(tmp1.name, "PNG")
            resize_with_padding(img_path, random.choice(scale_factors), tmp2.name, image_size=image_size)
            stitched_path = os.path.join(sanity_dir, f"{model_slug}_llava_stitched_example.png")
            stitch_two_images_with_labels(tmp1.name, tmp2.name, stitched_path)
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
            logger.info(f"LLaVA stitched sanity example saved to: {stitched_path} ({domain}/{obj})")
        except Exception as e:
            logger.warning(f"Failed to create LLaVA stitched sanity example: {e}")

    # Index for hard negatives: same domain, different object
    domain_to_entries: Dict[str, List[Tuple[str, str, str]]] = {}
    for domain, obj, img_path in pacs_images:
        domain_to_entries.setdefault(domain, []).append((domain, obj, img_path))

    results: List[Dict[str, any]] = []
    existing_keys = set()

    # Resume support
    csv_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion_pacs.csv")
    if resume_csv and os.path.exists(resume_csv):
        try:
            existing_df = pd.read_csv(resume_csv)
            results = existing_df.to_dict(orient="records")
            for _, row in existing_df.iterrows():
                scale_val = row.get("scale_factor")
                try:
                    scale_val = float(scale_val)
                except Exception:
                    continue
                key = (
                    row.get("dataset"),
                    row.get("domain"),
                    row.get("object"),
                    row.get("image_file"),
                    scale_val,
                    _normalize_bool(row.get("is_positive")),
                )
                existing_keys.add(key)
            msg = f"Resuming from {resume_csv} with {len(existing_keys)} existing entries"
            logger.info(msg)
            print(msg, flush=True)
        except Exception as e:
            logger.warning(f"Failed to load resume CSV ({resume_csv}): {e}")

    def append_result_row(row: Dict[str, any]) -> None:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    logger.info("Running PACS experiments...")
    pacs_permutations = [(domain, obj, img_path, scale) for (domain, obj, img_path) in pacs_images for scale in scale_factors]
    logger.info(f"Using all {len(pacs_permutations)} PACS samples (all images × all scales, both positive and negative)")

    for idx, (domain, obj, img_path, scale_factor) in enumerate(tqdm(pacs_permutations, desc="PACS")):
        image_file = os.path.basename(img_path)
        for is_positive in (True, False):
            key = ("pacs", domain, obj, image_file, float(scale_factor), is_positive)
            if key in existing_keys:
                continue

            # Image 1: Original (full size)
            img1 = Image.open(img_path).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
            temp_img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img1.close()
            img1.save(temp_img1.name, "PNG")
            img1_path = temp_img1.name

            # Image 2 selection
            if is_positive:
                img2_source = img_path
            else:
                candidates = [e for e in domain_to_entries.get(domain, []) if e[1] != obj]
                if not candidates:
                    candidates = [e for e in pacs_images if e[1] != obj]
                if not candidates:
                    os.unlink(temp_img1.name)
                    continue
                _, obj2, img2_source = random.choice(candidates)

            temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img2.close()
            resize_with_padding(img2_source, scale_factor, temp_img2.name, image_size=image_size)
            img2_path = temp_img2.name

            try:
                if is_llava:
                    stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    stitched_tmp.close()
                    stitch_two_images_with_labels(img1_path, img2_path, stitched_tmp.name)
                    prompt_for_model = (
                        "You are given a single composite image with two panels. "
                        'The left panel is labeled "image1" and the right panel is labeled "image2".\n\n'
                        + prompt_template
                    )
                    infer_payload = {"image_path": stitched_tmp.name, "text_prompt": prompt_for_model}
                else:
                    infer_payload = {
                        "image_paths": [img1_path, img2_path],
                        "text_prompt": prompt_template,
                        "max_pixels": image_size * image_size * 2,
                    }
                response = safe_infer(model, infer_payload)
                parsed = parse_response(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
            except Exception as e:
                logger.warning(f"Error in inference for PACS sample {idx}: {e}")
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    os.unlink(temp_img1.name)
                    os.unlink(temp_img2.name)
                    if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                        os.unlink(stitched_tmp.name)
                    continue
                os.unlink(temp_img1.name)
                os.unlink(temp_img2.name)
                if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                    os.unlink(stitched_tmp.name)
                continue

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            os.unlink(temp_img1.name)
            os.unlink(temp_img2.name)
            if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

            is_correct = (prediction == "yes" and is_positive) or (prediction == "no" and not is_positive)
            row = {
                "dataset": "pacs",
                "domain": domain,
                "object": obj,
                "image_file": image_file,
                "is_positive": is_positive,
                "scale_factor": float(scale_factor),
                "prediction": prediction,
                "is_correct": bool(is_correct),
                "response": response_clean,
            }
            results.append(row)
            append_result_row(row)

    shutil.rmtree(temp_dir)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # --- Analysis ---
    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS (PACS)")
    logger.info("=" * 80)
    overall_acc = df["is_correct"].mean() * 100 if len(df) else 0.0
    logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({int(df['is_correct'].sum())}/{len(df)})")

    pos_df = df[df["is_positive"] == True]
    neg_df = df[df["is_positive"] == False]
    tpr = (len(pos_df[pos_df["prediction"] == "yes"]) / len(pos_df) * 100) if len(pos_df) else 0.0
    tnr = (len(neg_df[neg_df["prediction"] == "no"]) / len(neg_df) * 100) if len(neg_df) else 0.0
    logger.info(f"Positive pairs (Recall/TPR): {tpr:.2f}% (n={len(pos_df)})")
    logger.info(f"Negative pairs (Specificity/TNR): {tnr:.2f}% (n={len(neg_df)})")

    # By domain summary
    domain_summary = (
        df.groupby("domain")["is_correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_samples", "sum": "n_correct"})
    )
    domain_summary["accuracy"] = domain_summary["accuracy"] * 100
    domain_summary_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion_pacs_domain_summary.csv")
    domain_summary.to_csv(domain_summary_path, index=False)
    logger.info(f"Domain summary saved to: {domain_summary_path}")

    # By scale summary
    scale_summary = (
        df.groupby("scale_factor")["is_correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_samples", "sum": "n_correct"})
        .sort_values("scale_factor")
    )
    scale_summary["accuracy"] = scale_summary["accuracy"] * 100
    scale_summary_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion_pacs_scale_summary.csv")
    scale_summary.to_csv(scale_summary_path, index=False)
    logger.info(f"Scale summary saved to: {scale_summary_path}")

    # Plot
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Overall
        axes[0].bar(["PACS"], [overall_acc], color="steelblue", edgecolor="black", alpha=0.8)
        axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Overall Accuracy")
        axes[0].text(0, overall_acc + 2, f"{overall_acc:.1f}%", ha="center", fontweight="bold")

        # By domain
        dom_sorted = domain_summary.sort_values("accuracy", ascending=False)
        axes[1].bar(dom_sorted["domain"], dom_sorted["accuracy"], color="seagreen", edgecolor="black", alpha=0.8)
        axes[1].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Accuracy by Domain")
        axes[1].tick_params(axis="x", rotation=45, labelsize=9)

        # By scale
        axes[2].bar(
            [f"{s:.1f}" for s in scale_summary["scale_factor"].tolist()],
            scale_summary["accuracy"].tolist(),
            color="coral",
            edgecolor="black",
            alpha=0.8,
        )
        axes[2].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[2].set_ylim(0, 100)
        axes[2].set_title("Accuracy by Scale Factor")
        axes[2].set_xlabel("Scale Factor")

        plt.tight_layout()
        plot_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion_pacs_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to: {plot_path}")
    except Exception as e:
        logger.warning(f"Failed to create plot: {e}")

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED (PACS)")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {exp_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Illusion Experiment (PACS)")
    parser.add_argument(
        "--pacs_dir",
        type=str,
        default=None, help="Path to PACS subset directory (required)",
        help="Directory containing PACS subset folders (e.g., art_painting_dog/...).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        choices=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "qwen2.5-vl",
            "gemini-2.5-pro",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "gpt-5.2",
            "Qwen/Qwen3-VL-235B-A22B-Thinking",
            "llava",
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen3-VL-8B-Instruct"
        ],
        help="Model to use",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="(Unused) kept for compatibility.")
    parser.add_argument("--positive_ratio", type=float, default=0.5, help="(Unused) fixed 50/50.")
    parser.add_argument("--use_cot_prompt", action="store_true", help="Use Chain-of-Thought prompt")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument("--image_size", type=int, default=336, help="Image size (default: 336)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0, help="Delay between API calls (seconds)")
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.9],
        help="Scale factors for experiment (default: 0.1 0.3 0.5 0.9)",
    )
    parser.add_argument("--resume_csv", type=str, default=None, help="Optional CSV to resume from (skip existing entries)")
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="direct",
        choices=["direct", "cot", "v2", "v3"],
        help="Prompt version to use (default: direct)",
    )
    parser.add_argument("--prompt_tag", type=str, default=None, help="Optional tag appended to output path")
    parser.add_argument(
        "--max_images_per_subcategory",
        type=int,
        default=None,
        help="Optional cap on number of images loaded per <domain>_<object> folder.",
    )

    args = parser.parse_args()

    model_config = {"model_name": args.model}
    run_scale_illusion_pacs_experiment(
        pacs_dir=args.pacs_dir,
        model_config=model_config,
        num_samples=args.num_samples,
        positive_ratio=args.positive_ratio,
        use_cot_prompt=args.use_cot_prompt,
        output_dir=args.output_dir,
        log_path=args.log_path,
        image_size=args.image_size,
        rate_limit_delay=args.rate_limit_delay,
        scale_factors=args.scale_factors,
        resume_csv=args.resume_csv,
        prompt_version=args.prompt_version,
        prompt_tag=args.prompt_tag,
        max_images_per_subcategory=args.max_images_per_subcategory,
    )

