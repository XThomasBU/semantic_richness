"""
Experiment: Identity Illusion (same image twice vs. different characters)

Protocol mirrors scale_illusion.py but **no transformation** on the second image:
- Positive pairs: Image A (normalized) vs. **the same** Image A (normalized).
- Negative pairs: Image A (normalized) vs. Image B (normalized), hard negatives from the same script.

Prompt:
  Are these the same characters? Answer in curly brackets, e.g. {Yes} or {No}
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.genai.errors import ServerError
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.infer import InferenceModel

PROMPT_IDENTITY = (
    "Are these the same characters? Answer in curly brackets, e.g. {Yes} or {No}"
)


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
    canvas_h = h + margin * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    x1, y1 = margin, margin
    x2, y2 = margin * 2 + w1, margin
    canvas.paste(img1, (x1, y1))
    canvas.paste(img2, (x2, y2))

    canvas.save(output_path, "PNG")
    return output_path


def setup_logging(log_path: str = None):
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if log_path:
        logging.basicConfig(filename=log_path, level=logging.INFO, format=FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT)
    return logging.getLogger(__name__)


def safe_infer(model, infer_args, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            return model.infer(infer_args)
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = (2**attempt) + random.random() * 2
                print(
                    f"[WARN] Server overloaded (attempt {attempt}/{max_retries}). Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + random.random()
                print(
                    f"[WARN] Error during inference (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time:.1f}s..."
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


def load_alphabet_images(alphabet_dir: str) -> List[Tuple[str, str, str]]:
    alphabet_path = Path(alphabet_dir) / "times_new_roman"
    if not alphabet_path.exists():
        raise FileNotFoundError(f"Alphabet directory not found: {alphabet_path}")

    images = []
    for char_dir in sorted(alphabet_path.glob("character*")):
        char_id = char_dir.name
        image_file = char_dir / "image.png"
        if image_file.exists():
            images.append((char_id, str(image_file), str(image_file)))

    return images


def load_omniglot_images(omniglot_dir: str, allowed_scripts: List[str] = None) -> List[Tuple[str, str, str, str]]:
    omniglot_path = Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
    if not omniglot_path.exists():
        raise FileNotFoundError(f"Omniglot directory not found: {omniglot_path}")

    images = []
    allowed_set = {s.lower() for s in allowed_scripts} if allowed_scripts else None
    for script_dir in sorted([p for p in omniglot_path.iterdir() if p.is_dir()]):
        script_name = script_dir.name
        if allowed_set is not None and script_name.lower() not in allowed_set:
            continue
        for char_dir in sorted(script_dir.glob("character*")):
            char_id = char_dir.name
            image_files = sorted(char_dir.glob("*.png"))
            if image_files:
                first_img = image_files[0]
                second_img = image_files[1] if len(image_files) > 1 else image_files[0]
                images.append((script_name, char_id, str(first_img), str(second_img)))

    return images


def _normalize_image_to_temp(path: str, image_size: int) -> str:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name, "PNG")
    return tmp.name


def _acc_tpr_tnr_from_subset(script_df: pd.DataFrame) -> Dict[str, float]:
    """Accuracy (%), TPR (%), TNR (%) plus confusion counts."""
    pos_df = script_df[script_df["is_positive"] == True]
    neg_df = script_df[script_df["is_positive"] == False]
    tp = len(pos_df[pos_df["prediction"] == "yes"])
    fn = len(pos_df) - tp
    tn = len(neg_df[neg_df["prediction"] == "no"])
    fp = len(neg_df) - tn
    tpr = (tp / len(pos_df) * 100) if len(pos_df) > 0 else float("nan")
    tnr = (tn / len(neg_df) * 100) if len(neg_df) > 0 else float("nan")
    acc = script_df["is_correct"].mean() * 100 if len(script_df) > 0 else float("nan")
    return {
        "accuracy": acc,
        "tpr": tpr,
        "tnr": tnr,
        "n_samples": len(script_df),
        "n_correct": int(script_df["is_correct"].sum()) if len(script_df) else 0,
        "n_positive": len(pos_df),
        "n_negative": len(neg_df),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def parse_response_identity(response_text: str) -> Dict[str, any]:
    """Parse model output for {Yes}/{No} style and plain YES/NO."""
    if not response_text:
        return {"answer": "unknown", "response_clean": ""}

    if isinstance(response_text, list):
        response_text = response_text[0] if response_text else ""

    response_clean = str(response_text)
    text_lower = response_clean.lower()

    # Curly-bracket answers first
    m = re.search(r"\{\s*(yes|no)\s*\}", text_lower)
    if m:
        return {"answer": "yes" if m.group(1) == "yes" else "no", "response_clean": response_clean}

    text_for_matching = text_lower.strip()
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
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def create_identity_illusion_sanity_check(
    alphabet_dir: str,
    omniglot_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
):
    alphabet_images = load_alphabet_images(alphabet_dir)
    omniglot_images = load_omniglot_images(omniglot_dir)

    if len(alphabet_images) < 2 or len(omniglot_images) < 2:
        raise ValueError("Not enough images for sanity check")

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 2.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        if i < num_examples // 2:
            char_id, img_path, _ = random.choice(alphabet_images)
            is_positive = i % 2 == 0
            img1 = Image.open(img_path).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
            if is_positive:
                img2 = img1.copy()
                label = f"Alphabet: Positive\n{char_id} (same image ×2)"
            else:
                char_id2, img_path2, _ = random.choice([x for x in alphabet_images if x[0] != char_id])
                img2 = Image.open(img_path2).convert("RGB")
                if img2.size != (image_size, image_size):
                    img2 = img2.resize((image_size, image_size), Image.Resampling.LANCZOS)
                label = f"Alphabet: Negative\n{char_id} vs {char_id2}"
        else:
            script_name, char_id, img_path_first, _ = random.choice(omniglot_images)
            is_positive = i % 2 == 0
            img1 = Image.open(img_path_first).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
            if is_positive:
                img2 = img1.copy()
                label = f"Omniglot: Positive\n{script_name}/{char_id} (same image ×2)"
            else:
                same_script = [(s, c, p1) for s, c, p1, _ in omniglot_images if s == script_name and c != char_id]
                if same_script:
                    _, char_id2, p2 = random.choice(same_script)
                else:
                    _, char_id2, p2, _ = random.choice([x for x in omniglot_images if x[1] != char_id])
                img2 = Image.open(p2).convert("RGB")
                if img2.size != (image_size, image_size):
                    img2 = img2.resize((image_size, image_size), Image.Resampling.LANCZOS)
                label = f"Omniglot: Negative\n{script_name}/{char_id} vs {char_id2}"

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
                "Image 1",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            axes[i, 1].text(
                0.5,
                -0.1,
                "Image 2\n(identity: same file)",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

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
    print(f"Identity illusion sanity check saved to: {output_path}")


def run_identity_illusion_experiment(
    alphabet_dir: str,
    omniglot_dir: str,
    model_config: Dict[str, str] = {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct"},
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    omniglot_scripts: List[str] = None,
    resume_csv: str = None,
    prompt_tag: str = None,
):
    logger = setup_logging(log_path)
    prompt_template = PROMPT_IDENTITY

    logger.info("=" * 80)
    logger.info("IDENTITY ILLUSION EXPERIMENT (same image vs different; no scale transform)")
    logger.info("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="identity_illusion_")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        model_name = model_config.get("model_name", "unknown")
        model = InferenceModel(model_name)
        model_slug = _model_slug(model_name)
        logger.info(f"Model initialized: {model_name}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise e

    alphabet_images = load_alphabet_images(alphabet_dir)
    if omniglot_scripts:
        omniglot_scripts = [s for s in omniglot_scripts if str(s).strip().lower() != "english"]
        logger.info(f"Limiting Omniglot scripts to: {len(omniglot_scripts)} names")
    else:
        logger.info("Using all Omniglot scripts under images_all")
    omniglot_images = load_omniglot_images(omniglot_dir, allowed_scripts=omniglot_scripts)

    if len(alphabet_images) < 2 or len(omniglot_images) < 2:
        raise ValueError("Need at least 2 alphabet and 2 Omniglot entries")

    exp_dir_name = f"identity_illusion_{model_slug}" if model_name == "gemini-2.5-pro" else "identity_illusion"
    if prompt_tag:
        exp_dir_name = f"{exp_dir_name}_{prompt_tag}"
    exp_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)

    try:
        sanity_path = os.path.join(sanity_dir, f"{model_slug}_identity_illusion_sanity_check.png")
        create_identity_illusion_sanity_check(
            alphabet_dir=alphabet_dir,
            omniglot_dir=omniglot_dir,
            output_path=sanity_path,
            num_examples=6,
            image_size=image_size,
        )
    except Exception as e:
        logger.error(f"Failed sanity check: {e}")
        raise e

    is_llava = model_name in ("llava", "llava-1.5-7b") or str(model_name).startswith("llava-hf/")

    results = []
    existing_keys = set()
    if resume_csv and os.path.exists(resume_csv):
        try:
            existing_df = pd.read_csv(resume_csv)
            results = existing_df.to_dict(orient="records")
            for _, row in existing_df.iterrows():
                key = (
                    row.get("dataset"),
                    row.get("script_name"),
                    row.get("char_id"),
                    _normalize_bool(row.get("is_positive")),
                )
                existing_keys.add(key)
            logger.info(f"Resuming from {resume_csv} with {len(existing_keys)} existing entries")
            print(f"Resuming from {resume_csv} with {len(existing_keys)} existing entries", flush=True)
        except Exception as e:
            logger.warning(f"Failed to load resume CSV: {e}")

    csv_path = os.path.join(exp_dir, f"{model_slug}_identity_illusion.csv")

    def append_result_row(row: Dict[str, any]) -> None:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # Alphabet: each character × (positive, negative); second image is same file content (identity) or other char
    alphabet_samples = [(c, p1, p2) for c, p1, p2 in alphabet_images]

    for idx, (char_id, img_path_first, img_path_second) in enumerate(tqdm(alphabet_samples, desc="Alphabet")):
        for is_positive in (True, False):
            if ("alphabet", "English", char_id, is_positive) in existing_keys:
                continue

            img1_path = _normalize_image_to_temp(img_path_first, image_size)
            if is_positive:
                # Second panel: same character, same pixels (no transform)
                img2_path = _normalize_image_to_temp(img_path_first, image_size)
            else:
                _, other_first, _ = random.choice([x for x in alphabet_images if x[0] != char_id])
                img2_path = _normalize_image_to_temp(other_first, image_size)

            try:
                if is_llava:
                    stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    stitched_tmp.close()
                    stitch_two_images_with_labels(img1_path, img2_path, stitched_tmp.name)
                    infer_payload = {"image_path": stitched_tmp.name, "text_prompt": prompt_template}
                else:
                    infer_payload = {
                        "image_paths": [img1_path, img2_path],
                        "text_prompt": prompt_template,
                        "max_pixels": image_size * image_size * 2,
                    }
                response = safe_infer(model, infer_payload)
                parsed = parse_response_identity(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
            except Exception as e:
                logger.warning(f"Alphabet sample {idx}: {e}")
                os.unlink(img1_path)
                if os.path.exists(img2_path):
                    os.unlink(img2_path)
                if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                    os.unlink(stitched_tmp.name)
                continue

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            os.unlink(img1_path)
            os.unlink(img2_path)
            if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

            is_correct = (prediction == "yes" and is_positive) or (prediction == "no" and not is_positive)
            row = {
                "dataset": "alphabet",
                "script_name": "English",
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": 1.0,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
            results.append(row)
            append_result_row(row)

    omniglot_samples = list(omniglot_images)

    for idx, (script_name, char_id, img_path_first, img_path_second) in enumerate(tqdm(omniglot_samples, desc="Omniglot")):
        for is_positive in (True, False):
            if ("omniglot", script_name, char_id, is_positive) in existing_keys:
                continue

            img1_path = _normalize_image_to_temp(img_path_first, image_size)
            if is_positive:
                img2_path = _normalize_image_to_temp(img_path_first, image_size)
            else:
                same_script = [(s, c, p1) for s, c, p1, _ in omniglot_images if s == script_name and c != char_id]
                if same_script:
                    _, _, other_p = random.choice(same_script)
                else:
                    _, _, other_p, _ = random.choice([x for x in omniglot_images if x[1] != char_id])
                img2_path = _normalize_image_to_temp(other_p, image_size)

            try:
                if is_llava:
                    stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    stitched_tmp.close()
                    stitch_two_images_with_labels(img1_path, img2_path, stitched_tmp.name)
                    infer_payload = {"image_path": stitched_tmp.name, "text_prompt": prompt_template}
                else:
                    infer_payload = {
                        "image_paths": [img1_path, img2_path],
                        "text_prompt": prompt_template,
                        "max_pixels": image_size * image_size * 2,
                    }
                response = safe_infer(model, infer_payload)
                parsed = parse_response_identity(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
            except Exception as e:
                logger.warning(f"Omniglot sample {idx}: {e}")
                os.unlink(img1_path)
                if os.path.exists(img2_path):
                    os.unlink(img2_path)
                if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                    os.unlink(stitched_tmp.name)
                continue

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            os.unlink(img1_path)
            os.unlink(img2_path)
            if is_llava and "stitched_tmp" in locals() and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

            is_correct = (prediction == "yes" and is_positive) or (prediction == "no" and not is_positive)
            row = {
                "dataset": "omniglot",
                "script_name": script_name,
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": 1.0,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
            results.append(row)
            append_result_row(row)

    shutil.rmtree(temp_dir, ignore_errors=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    logger.info(f"Results saved to: {csv_path}")

    alphabet_df = df[df["dataset"] == "alphabet"]
    omniglot_df = df[df["dataset"] == "omniglot"]
    alphabet_acc = alphabet_df["is_correct"].mean() * 100 if len(alphabet_df) else 0
    omniglot_acc = omniglot_df["is_correct"].mean() * 100 if len(omniglot_df) else 0
    logger.info(f"Alphabet accuracy: {alphabet_acc:.2f}%")
    logger.info(f"Omniglot accuracy: {omniglot_acc:.2f}%")

    script_summary_rows = []
    for script_name in sorted(df["script_name"].dropna().unique()):
        sdf = df[df["script_name"] == script_name]
        m = _acc_tpr_tnr_from_subset(sdf)
        script_summary_rows.append({"script_name": script_name, **m})
    summary_path = os.path.join(exp_dir, f"{model_slug}_identity_illusion_script_summary.csv")
    pd.DataFrame(script_summary_rows).to_csv(summary_path, index=False)
    logger.info(f"Per-script acc/TPR/TNR saved to: {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, dfg) in zip(axes, [("Alphabet", alphabet_df), ("Omniglot", omniglot_df)]):
        pos = dfg[dfg["is_positive"] == True]
        neg = dfg[dfg["is_positive"] == False]
        tpr = (len(pos[pos["prediction"] == "yes"]) / len(pos) * 100) if len(pos) else 0
        tnr = (len(neg[neg["prediction"] == "no"]) / len(neg) * 100) if len(neg) else 0
        ax.bar(["Positive (TPR)", "Negative (TNR)"], [tpr, tnr], color=["steelblue", "coral"], edgecolor="black")
        ax.set_ylim(0, 100)
        ax.set_title(f"{name} (n={len(dfg)})")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.6)
        ax.set_ylabel("Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{model_slug}_identity_illusion_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Experiment directory: {exp_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identity Illusion Experiment (same image, no scale transform)")
    parser.add_argument("--alphabet_dir", type=str, required=True)
    parser.add_argument("--omniglot_dir", type=str, required=True)
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
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "llava",
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
            "Qwen/Qwen3-VL-8B-Instruct",
        ],
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--rate_limit_delay", type=float, default=0.0)
    parser.add_argument(
        "--omniglot_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Optional script folder names to include; if omitted, use all scripts under images_all.",
    )
    parser.add_argument("--resume_csv", type=str, default=None)
    parser.add_argument("--prompt_tag", type=str, default=None)

    args = parser.parse_args()
    run_identity_illusion_experiment(
        alphabet_dir=args.alphabet_dir,
        omniglot_dir=args.omniglot_dir,
        model_config={"model_name": args.model},
        output_dir=args.output_dir,
        log_path=args.log_path,
        image_size=args.image_size,
        rate_limit_delay=args.rate_limit_delay,
        omniglot_scripts=args.omniglot_scripts if args.omniglot_scripts else None,
        resume_csv=args.resume_csv,
        prompt_tag=args.prompt_tag,
    )
