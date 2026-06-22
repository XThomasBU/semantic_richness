"""
Spatial Illusion (PACS): Rotation invariance vs. semantic familiarity on PACS domains.

Uses zmeurer's PACS rotation protocol (pac/rotation_recog.py):
  - Data: pass via --data_dir argument
  - Sample 200 images per domain (balanced across 7 classes) via flwrlabs/pacs
  - Positive: image X vs image X rotated by angle (PIL rotate, expand=False)
  - Negative (hard): image X vs image Y (different image, not rotated)
  - Domains: art_painting, cartoon, photo, sketch

Inference stack and analysis follow experiments/identity/spatial_illusion.py.
"""

import csv
import os
import sys
import argparse
import logging
import random
import re
import time
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.infer import InferenceModel
from google.genai.errors import ServerError

# zmeurer PACS paths / protocol defaults
DEFAULT_PACS_DATA_DIR = None
PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
SAMPLES_PER_DOMAIN = 200
ROTATION_ANGLES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

LABEL_ID_TO_NAME = {
    0: "dog",
    1: "elephant",
    2: "giraffe",
    3: "guitar",
    4: "horse",
    5: "house",
    6: "person",
}

PROMPT_COT = """Look at the following two images.

Do these two images show the same underlying object? One could be a rotated version of the other.

Answer with just "YES" or "NO" after your reasoning."""

PROMPT_DIRECT = (
    "If I rotate the first image, can I get the second image? "
    "Answer in curly brackets, e.g. {Yes} or {No}."
)

PROMPT_V2 = """Compare the two images and decide if they show the same object.
Ignore differences in rotation. Answer with exactly YES or NO."""

PROMPT_V3 = """Do the two images show the same visual object, even if one is rotated? Answer YES or NO."""


def stitch_two_images_with_labels(
    image1_path: str,
    image2_path: str,
    output_path: str,
    margin: int = 12,
    bg_color=(255, 255, 255),
):
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    w1, h1 = img1.size
    w2, h2 = img2.size
    h = max(h1, h2)

    if h1 != h:
        canvas1 = Image.new("RGB", (w1, h), bg_color)
        canvas1.paste(img1, (0, (h - h1) // 2))
        img1 = canvas1
        w1, h1 = img1.size
    if h2 != h:
        canvas2 = Image.new("RGB", (w2, h), bg_color)
        canvas2.paste(img2, (0, (h - h2) // 2))
        img2 = canvas2
        w2, h2 = img2.size

    canvas_w = w1 + w2 + margin * 3
    canvas_h = h + margin * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    canvas.paste(img1, (margin, margin))
    canvas.paste(img2, (margin * 2 + w1, margin))
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
    for attempt in range(1, max_retries + 1):
        try:
            return model.infer(infer_args)
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = (2**attempt) + random.random() * 2
                print(
                    f"[WARN] Server overloaded (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + random.random()
                print(
                    f"[WARN] Error during inference (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Model unavailable after {max_retries} retries.")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_response(response_text: str) -> Dict[str, Any]:
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


def list_pacs_image_paths(data_dir: str) -> List[str]:
    """Same global sort order as zmeurer pac/rotation_recog.py."""
    names = sorted(
        [n for n in os.listdir(data_dir) if n.lower().endswith(".png")],
        key=lambda x: int(re.search(r"_\d+\.png", x).group().strip("_.png")),
    )
    return [os.path.join(data_dir, n) for n in names]


def parse_pacs_label(image_path: str) -> int:
    return int(re.search(r"_\d_", os.path.basename(image_path)).group().strip("_"))


def balanced_counts(total: int, n_classes: int) -> List[int]:
    base = total // n_classes
    remainder = total % n_classes
    counts = [base] * n_classes
    for i in range(remainder):
        counts[i] += 1
    return counts


def sample_pacs_indices_for_domain(
    domain: str,
    samples_per_domain: int = SAMPLES_PER_DOMAIN,
    seed: int = 42,
) -> List[int]:
    """
    Replicate zmeurer rotation_recog HF sampling: balanced classes per domain.
    Returns global indices into list_pacs_image_paths(data_dir).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("flwrlabs/pacs", split="train")
    classes = sorted(set(ds["label"]))

    grouped_indices = defaultdict(list)
    for idx, example in enumerate(ds):
        grouped_indices[(example["domain"], example["label"])].append(idx)

    counts = balanced_counts(samples_per_domain, len(classes))
    selected: List[int] = []
    for label, n in zip(classes, counts):
        indices = grouped_indices[(domain, label)]
        n = min(n, len(indices))
        selected.extend(rng.sample(indices, n))
    return selected


def pick_negative_index(image_idx: int, num_images: int, rng: random.Random) -> int:
    """Same distractor logic as rotation_recog.py (one Y per anchor X)."""
    counter = 0
    r = rng.randint(0, num_images - 1)
    while r == image_idx and counter < 5:
        r = rng.randint(0, num_images - 1)
        counter += 1
    return r


def save_rotated_image(image_path: str, angle: float, output_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    rotated = img.rotate(angle, expand=False)
    rotated.save(output_path, "PNG")
    return output_path


def pacs_rotation_image_paths(
    anchor_path: str,
    other_path: str,
    angle: float,
    is_positive: bool,
    temp_dir: str,
) -> Tuple[str, str]:
    """
    Positive: [X, rotate(X, angle)]; negative: [X, Y] (Y not rotated), matching rotation_recog.py.
    Returns paths to temporary files (caller should delete after inference).
    """
    img1_tmp = tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=temp_dir, prefix="pacs_img1_"
    )
    img1_tmp.close()
    Image.open(anchor_path).convert("RGB").save(img1_tmp.name, "PNG")

    if is_positive:
        img2_tmp = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, dir=temp_dir, prefix="pacs_rot_"
        )
        img2_tmp.close()
        save_rotated_image(anchor_path, angle, img2_tmp.name)
        return img1_tmp.name, img2_tmp.name

    img2_tmp = tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=temp_dir, prefix="pacs_other_"
    )
    img2_tmp.close()
    Image.open(other_path).convert("RGB").save(img2_tmp.name, "PNG")
    return img1_tmp.name, img2_tmp.name


def run_spatial_illusion_pacs_experiment(
    data_dir: str = DEFAULT_PACS_DATA_DIR,
    model_config: Dict[str, str] = None,
    domains: Optional[List[str]] = None,
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    rotation_angles: Optional[List[float]] = None,
    samples_per_domain: int = SAMPLES_PER_DOMAIN,
    resume_csv: str = None,
    prompt_version: str = "direct",
    prompt_tag: str = None,
    use_cot_prompt: bool = False,
    seed: int = 42,
):
    if model_config is None:
        model_config = {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct"}
    if rotation_angles is None:
        rotation_angles = list(ROTATION_ANGLES)
    if domains is None:
        domains = list(PACS_DOMAINS)

    logger = setup_logging(log_path)
    random.seed(seed)

    logger.info("=" * 80)
    logger.info("SPATIAL ILLUSION (PACS ROTATION) EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"PACS data dir (zmeurer): {data_dir}")
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info(f"Domains: {domains}")
    logger.info(f"Samples per domain: {samples_per_domain}")
    logger.info(f"Rotation angles: {rotation_angles}")
    logger.info(f"Output directory: {output_dir}")

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"PACS data directory not found: {data_dir}")

    logger.info("Initializing model...")
    model_name = model_config.get("model_name", "unknown")
    model = InferenceModel(model_name)
    model_slug = _model_slug(model_name)
    logger.info(f"Model initialized: {model_name}")

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

    exp_dir_name = (
        f"spatial_illusion_pacs_{model_slug}"
        if model_name == "gemini-2.5-pro"
        else "spatial_illusion_pacs"
    )
    if resolved_tag:
        exp_dir_name = f"{exp_dir_name}_{resolved_tag}"
    exp_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")

    image_paths = list_pacs_image_paths(data_dir)
    num_images = len(image_paths)
    if num_images < 2:
        raise ValueError(f"Need at least 2 images in {data_dir}, found {num_images}")
    logger.info(f"Loaded {num_images} PACS image paths from data dir")

    is_llava = model_name in ("llava", "llava-1.5-7b") or str(model_name).startswith(
        "llava-hf/"
    )

    existing_keys = set()
    if resume_csv and os.path.exists(resume_csv):
        try:
            existing_df = pd.read_csv(resume_csv)
            for _, row in existing_df.iterrows():
                try:
                    angle_val = float(row.get("angle"))
                except Exception:
                    continue
                existing_keys.add(
                    (
                        row.get("domain"),
                        int(row.get("image_id")),
                        angle_val,
                        _normalize_bool(row.get("is_positive")),
                    )
                )
            logger.info(
                f"Resuming from {resume_csv} with {len(existing_keys)} existing entries"
            )
        except Exception as e:
            logger.warning(f"Failed to load resume CSV ({resume_csv}): {e}")

    csv_path = os.path.join(exp_dir, f"{model_slug}_spatial_illusion_pacs.csv")

    def append_result_row(row: Dict[str, Any]) -> None:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    temp_dir = tempfile.mkdtemp(prefix="spatial_illusion_pacs_")
    rng = random.Random(seed)
    saved_sanity = False

    for domain in domains:
        logger.info(f"Sampling PACS indices for domain={domain}...")
        selected = sample_pacs_indices_for_domain(
            domain, samples_per_domain=samples_per_domain, seed=seed
        )
        selected_paths = [image_paths[idx] for idx in selected]
        logger.info(f"Domain {domain}: {len(selected_paths)} images")

        for image_idx, image_path in enumerate(
            tqdm(selected_paths, desc=f"PACS {domain}")
        ):
            label_id = parse_pacs_label(image_path)
            label_name = LABEL_ID_TO_NAME.get(label_id, str(label_id))
            global_idx = selected[image_idx]
            neg_idx = pick_negative_index(global_idx, num_images, rng)
            other_path = image_paths[neg_idx]

            for angle in rotation_angles:
                for is_positive in (True, False):
                    key = (domain, image_idx, float(angle), is_positive)
                    if key in existing_keys:
                        continue

                    img1_path, img2_path = pacs_rotation_image_paths(
                        image_path,
                        other_path,
                        float(angle),
                        is_positive,
                        temp_dir,
                    )

                    stitched_tmp = None
                    try:
                        if is_llava:
                            stitched_tmp = tempfile.NamedTemporaryFile(
                                suffix=".png", delete=False
                            )
                            stitched_tmp.close()
                            stitch_two_images_with_labels(
                                img1_path, img2_path, stitched_tmp.name
                            )
                            infer_payload = {
                                "image_path": stitched_tmp.name,
                                "text_prompt": prompt_template,
                            }
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
                        logger.warning(
                            f"Inference failed {domain}/{label_name}/"
                            f"img{image_idx}/angle={angle}/pos={is_positive}: {e}"
                        )
                        continue
                    finally:
                        for p in (img1_path, img2_path):
                            if os.path.exists(p):
                                os.unlink(p)
                        if stitched_tmp and os.path.exists(stitched_tmp.name):
                            os.unlink(stitched_tmp.name)

                    if rate_limit_delay > 0:
                        time.sleep(rate_limit_delay)

                    is_correct = (prediction == "yes" and is_positive) or (
                        prediction == "no" and not is_positive
                    )
                    row = {
                        "dataset": "pacs",
                        "domain": domain,
                        "label": label_name,
                        "label_id": label_id,
                        "image_id": image_idx,
                        "global_index": global_idx,
                        "image_file": os.path.basename(image_path),
                        "angle": float(angle),
                        "is_positive": is_positive,
                        "ground_truth": "yes" if is_positive else "no",
                        "prediction": prediction,
                        "is_correct": bool(is_correct),
                        "response": response_clean,
                    }
                    if not saved_sanity and is_positive:
                        try:
                            sanity_path = os.path.join(
                                sanity_dir,
                                f"{model_slug}_spatial_illusion_pacs_example.png",
                            )
                            stitch_two_images_with_labels(
                                img1_path, img2_path, sanity_path
                            )
                            saved_sanity = True
                        except Exception:
                            pass

                    append_result_row(row)

    try:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    logger.info(f"Results saved to: {csv_path}")

    if len(df) == 0:
        logger.warning("No results to analyze.")
        return df

    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS (PACS ROTATION)")
    logger.info("=" * 80)

    overall_acc = df["is_correct"].mean() * 100
    logger.info(
        f"Overall Accuracy: {overall_acc:.2f}% "
        f"({int(df['is_correct'].sum())}/{len(df)})"
    )

    pos_df = df[df["is_positive"] == True]
    neg_df = df[df["is_positive"] == False]
    tpr = (
        len(pos_df[pos_df["prediction"] == "yes"]) / len(pos_df) * 100
        if len(pos_df)
        else 0.0
    )
    tnr = (
        len(neg_df[neg_df["prediction"] == "no"]) / len(neg_df) * 100
        if len(neg_df)
        else 0.0
    )
    logger.info(f"Positive pairs (TPR): {tpr:.2f}% (n={len(pos_df)})")
    logger.info(f"Negative pairs (TNR): {tnr:.2f}% (n={len(neg_df)})")

    domain_summary = (
        df.groupby("domain")["is_correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_samples", "sum": "n_correct"})
    )
    domain_summary["accuracy"] = domain_summary["accuracy"] * 100
    domain_summary_path = os.path.join(
        exp_dir, f"{model_slug}_spatial_illusion_pacs_domain_summary.csv"
    )
    domain_summary.to_csv(domain_summary_path, index=False)

    angle_summary = (
        df.groupby("angle")["is_correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_samples", "sum": "n_correct"})
        .sort_values("angle")
    )
    angle_summary["accuracy"] = angle_summary["accuracy"] * 100
    angle_summary_path = os.path.join(
        exp_dir, f"{model_slug}_spatial_illusion_pacs_angle_summary.csv"
    )
    angle_summary.to_csv(angle_summary_path, index=False)

    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(
            ["PACS"], [overall_acc], color="steelblue", edgecolor="black", alpha=0.8
        )
        axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Overall Accuracy")

        dom_sorted = domain_summary.sort_values("accuracy", ascending=False)
        axes[1].bar(
            dom_sorted["domain"],
            dom_sorted["accuracy"],
            color="seagreen",
            edgecolor="black",
            alpha=0.8,
        )
        axes[1].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Accuracy by Domain")
        axes[1].tick_params(axis="x", rotation=45, labelsize=9)

        axes[2].bar(
            [str(int(a)) for a in angle_summary["angle"]],
            angle_summary["accuracy"],
            color="coral",
            edgecolor="black",
            alpha=0.8,
        )
        axes[2].axhline(50, color="gray", linestyle="--", linewidth=1)
        axes[2].set_ylim(0, 100)
        axes[2].set_title("Accuracy by Rotation Angle")
        axes[2].set_xlabel("Angle (degrees)")

        plt.tight_layout()
        plot_path = os.path.join(
            exp_dir, f"{model_slug}_spatial_illusion_pacs_plot.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to: {plot_path}")
    except Exception as e:
        logger.warning(f"Failed to create plot: {e}")

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED (PACS ROTATION)")
    logger.info("=" * 80)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial Illusion rotation experiment on PACS (zmeurer protocol)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_PACS_DATA_DIR,
        help="Directory with PACS PNGs (zmeurer pac/data).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        nargs="*",
        default=None,
        help="One or more PACS domains (default: all four).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="Model name passed to InferenceModel",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--rate_limit_delay", type=float, default=0.0)
    parser.add_argument(
        "--rotation_angles",
        type=float,
        nargs="+",
        default=None,
        help=f"Rotation angles in degrees (default: {ROTATION_ANGLES})",
    )
    parser.add_argument(
        "--samples_per_domain",
        type=int,
        default=SAMPLES_PER_DOMAIN,
        help="Balanced samples per domain (default: 200).",
    )
    parser.add_argument("--resume_csv", type=str, default=None)
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="direct",
        choices=["direct", "cot", "v2", "v3"],
    )
    parser.add_argument("--prompt_tag", type=str, default=None)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    domains = args.domain if args.domain else None

    run_spatial_illusion_pacs_experiment(
        data_dir=args.data_dir,
        model_config={"model_name": args.model},
        domains=domains,
        output_dir=args.output_dir,
        log_path=args.log_path,
        image_size=args.image_size,
        rate_limit_delay=args.rate_limit_delay,
        rotation_angles=args.rotation_angles,
        samples_per_domain=args.samples_per_domain,
        resume_csv=args.resume_csv,
        prompt_version=args.prompt_version,
        prompt_tag=args.prompt_tag,
        use_cot_prompt=args.use_cot_prompt,
        seed=args.seed,
    )
