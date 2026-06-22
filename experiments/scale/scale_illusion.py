"""
Experiment: The "Scale Illusion" (Scale Invariance vs. Semantic Familiarity)

Hypothesis: VLMs rely on semantic familiarity rather than scale-invariant geometric reasoning
when matching images at different sizes. They memorize canonical sizes of familiar objects
(like Latin letters) but fail to generalize to unfamiliar shapes (like Omniglot characters).

The Generalization Gap (Δ):
- Δ = Accuracy_Latin - Accuracy_Omniglot
- If Δ > 20%: Proof that the model is blind to scale and only bound to semantics
- If Δ ≈ 0: The model truly understands scale invariance (unlikely)

Protocol:
- Group A (High Semantic Prior): Standard Latin Alphabets (A, B, C...)
- Group B (Low Semantic Prior): Omniglot characters from rare alphabets
- Positive Pairs: Image X (full size) vs. Image X (scaled with padding)
- Negative Pairs: Image X (full size) vs. Image Y (scaled with padding) - Hard negatives from same script
- Scale Factors: Different scale factors (0.1, 0.3, 0.5, 0.7, 0.9) to test scale invariance
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
import pandas as pd
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.infer import InferenceModel
from google.genai.errors import ServerError

# --- CONFIGURATION ---
# Scale factors to test (0.0-1.0, where 1.0 = full size, 0.5 = half size)
SCALE_FACTORS = [0.1, 0.3, 0.5, 0.9]

# Latin/times_new_roman alphabet is always run separately as dataset "alphabet" / script "English".
# Omniglot: omit --omniglot_scripts to use every folder under images_all.

# Prompt options
PROMPT_COT = """Look at the following two images.

Do these two images show the same underlying character? One could be a scaled version of the other.

Answer with just "YES" or "NO" after your reasoning."""

# PROMPT_DIRECT = """Do these two images show the same character? One could be a scaled version of the other. Answer YES or NO."""
PROMPT_DIRECT = """You are given two images, each containing a single character on a white background. Decide whether they depict the same underlying character, allowing for significant differences in size, scale, or resolution. Answer with exactly YES or NO."""

# Alternative prompt version
PROMPT_V2 = """Compare the two images and decide if they show the same character.
Ignore differences in scale, size, or resolution. Answer with exactly YES or NO."""
PROMPT_V3 = """Do the two images show the same visual character, even if their size or resolution differs? Answer YES or NO."""


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
    """Setup logging configuration."""
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if log_path:
        logging.basicConfig(filename=log_path, level=logging.INFO, format=FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT)
    return logging.getLogger(__name__)


def safe_infer(model, infer_args, max_retries=5):
    """
    Run model inference with automatic retry on 503 UNAVAILABLE errors.
    Implements exponential backoff to handle overloaded servers.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return model.infer(infer_args)
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = (2**attempt) + random.random() * 2  # jitter
                print(
                    f"[WARN] Server overloaded (attempt {attempt}/{max_retries}). Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            # For other errors, log and retry once
            if attempt < max_retries:
                wait_time = 1 + random.random()
                print(
                    f"[WARN] Error during inference (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(
        f"Model unavailable after {max_retries} retries. Aborting inference."
    )


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_alphabet_images(alphabet_dir: str) -> List[Tuple[str, str, str]]:
    """
    Load alphabet images from times_new_roman directory.

    Returns:
        List of (character_id, image_path_first, image_path_second) tuples
    """
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


def load_omniglot_images(
    omniglot_dir: str, allowed_scripts: List[str] = None
) -> List[Tuple[str, str, str, str]]:
    """
    Load Omniglot images from the specified directory.
    Uses all scripts under images_all, taking only the first image
    under each character directory.

    Args:
        omniglot_dir: Base directory containing omniglot

    Returns:
        List of (script_name, character_id, image_path_first, image_path_second) tuples
    """
    omniglot_path = (
        Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
    )
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
            # Get first and second image from this character
            image_files = sorted(char_dir.glob("*.png"))
            if image_files:
                first_img = image_files[0]
                second_img = image_files[1] if len(image_files) > 1 else image_files[0]
                images.append((script_name, char_id, str(first_img), str(second_img)))

    return images


def resize_with_padding(
    image_path: str, scale_factor: float, output_path: str = None, image_size: int = 336
) -> str:
    """
    Resize an image by adding white padding, making the character smaller within the same canvas.

    Args:
        image_path: Path to input image
        scale_factor: Scale factor (0.0-1.0), where 1.0 = original size, 0.5 = half size
        output_path: Path to save resized image (if None, uses temp file)
        image_size: Target image size (canvas size)

    Returns:
        Path to resized image (same canvas size, but character is smaller)
    """
    img = Image.open(image_path).convert("RGB")

    # Resize to target size if needed (preprocessing)
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # Calculate new size for the character (smaller)
    new_char_size = int(image_size * scale_factor)

    # Resize the character to be smaller
    img_small = img.resize((new_char_size, new_char_size), Image.Resampling.LANCZOS)

    # Create a white canvas of the original size
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))

    # Paste the smaller character in the center
    paste_x = (image_size - new_char_size) // 2
    paste_y = (image_size - new_char_size) // 2
    canvas.paste(img_small, (paste_x, paste_y))

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"resized_{random.randint(0, 999999)}.png")

    canvas.save(output_path, "PNG")
    return output_path


def create_scale_illusion_sanity_check(
    alphabet_dir: str,
    omniglot_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
    scale_factors: List[float] = [0.1, 0.3, 0.5, 0.9],
):
    """
    Create sanity check visualization for scale experiment.
    Shows 2 columns: original (full size) vs scaled (with padding).
    """
    alphabet_images = load_alphabet_images(alphabet_dir)
    omniglot_images = load_omniglot_images(omniglot_dir)

    if len(alphabet_images) < 2 or len(omniglot_images) < 2:
        raise ValueError("Not enough images for sanity check")

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 2.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        scale_factor = random.choice(scale_factors)

        if i < num_examples // 2:
            # Alphabet examples
            char_id, img_path_first, img_path_second = random.choice(alphabet_images)
            is_positive = i % 2 == 0

            # Original image (full size)
            img1 = Image.open(img_path_first).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

            if is_positive:
                # Positive pair: same character, scaled
                temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_scaled.close()
                resize_with_padding(
                    img_path_second,
                    scale_factor,
                    temp_scaled.name,
                    image_size=image_size,
                )
                img2 = Image.open(temp_scaled.name).convert("RGB")
                os.unlink(temp_scaled.name)
                label = f"Alphabet: Positive\n{char_id} (scale={scale_factor:.1f})"
            else:
                # Negative pair: different character
                char_id2, img_path2_first, _ = random.choice(
                    [x for x in alphabet_images if x[0] != char_id]
                )
                temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_scaled.close()
                resize_with_padding(
                    img_path2_first,
                    scale_factor,
                    temp_scaled.name,
                    image_size=image_size,
                )
                img2 = Image.open(temp_scaled.name).convert("RGB")
                os.unlink(temp_scaled.name)
                label = f"Alphabet: Negative\n{char_id} vs {char_id2} (scale={scale_factor:.1f})"

            dataset = "Alphabet"
        else:
            # Omniglot examples
            script_name, char_id, img_path_first, img_path_second = random.choice(
                omniglot_images
            )
            is_positive = i % 2 == 0

            # Original image (full size)
            img1 = Image.open(img_path_first).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

            if is_positive:
                # Positive pair: same character, scaled
                temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_scaled.close()
                resize_with_padding(
                    img_path_second,
                    scale_factor,
                    temp_scaled.name,
                    image_size=image_size,
                )
                img2 = Image.open(temp_scaled.name).convert("RGB")
                os.unlink(temp_scaled.name)
                label = f"Omniglot: Positive\n{script_name}/{char_id} (scale={scale_factor:.1f})"
            else:
                # Negative pair: different character
                same_script = [
                    (s, c, p1, p2)
                    for s, c, p1, p2 in omniglot_images
                    if s == script_name and c != char_id
                ]
                if same_script:
                    script_name2, char_id2, img_path2_first, _ = random.choice(
                        same_script
                    )
                else:
                    script_name2, char_id2, img_path2_first, _ = random.choice(
                        [x for x in omniglot_images if x[1] != char_id]
                    )

                temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_scaled.close()
                resize_with_padding(
                    img_path2_first,
                    scale_factor,
                    temp_scaled.name,
                    image_size=image_size,
                )
                img2 = Image.open(temp_scaled.name).convert("RGB")
                os.unlink(temp_scaled.name)
                label = f"Omniglot: Negative\n{script_name}/{char_id} vs {script_name2}/{char_id2} (scale={scale_factor:.1f})"

            dataset = "Omniglot"

        # Resize for display
        img1_display = img1.resize((150, 150), Image.Resampling.LANCZOS)
        img2_display = img2.resize((150, 150), Image.Resampling.LANCZOS)

        # Column 1: Original (full size)
        axes[i, 0].imshow(img1_display)
        axes[i, 0].axis("off")

        # Column 2: Scaled (with padding)
        axes[i, 1].imshow(img2_display)
        axes[i, 1].axis("off")

        # Add label below the row
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

        # Add dataset and pair type label on the right
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
    print(f"Scale illusion sanity check visualization saved to: {output_path}")


def _acc_tpr_tnr_from_subset(script_df: pd.DataFrame) -> Dict[str, float]:
    """Accuracy (%), TPR (%), TNR (%) plus confusion counts for rows with is_positive / prediction."""
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


def parse_response(response_text: str) -> Dict[str, any]:
    """
    Parse model response to extract YES/NO answer.
    Preserves original response without stripping.
    """
    if not response_text:
        return {"answer": "unknown", "response_clean": ""}

    # Handle list responses (convert to string)
    if isinstance(response_text, list):
        response_text = response_text[0] if response_text else ""

    # Preserve original response exactly as-is
    response_clean = str(response_text)

    # Create a copy for matching (case-insensitive, stripped)
    text_for_matching = str(response_text).lower().strip()

    # Try to find YES or NO
    yes_patterns = [
        r"\byes\b",
        r"\btrue\b",
        r"\bsame\b",
    ]
    no_patterns = [
        r"\bno\b",
        r"\bfalse\b",
        r"\bdifferent\b",
    ]

    for pattern in yes_patterns:
        if re.search(pattern, text_for_matching):
            return {"answer": "yes", "response_clean": response_clean}

    for pattern in no_patterns:
        if re.search(pattern, text_for_matching):
            return {"answer": "no", "response_clean": response_clean}

    return {"answer": "unknown", "response_clean": response_clean}


def _model_slug(model_name: str) -> str:
    """Create a filesystem-safe model slug for output filenames."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def run_scale_illusion_experiment(
    alphabet_dir: str,
    omniglot_dir: str,
    model_config: Dict[str, str] = {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct"},
    num_samples: int = 50,
    positive_ratio: float = 0.5,
    use_cot_prompt: bool = False,
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    scale_factors: List[float] = [0.1, 0.3, 0.5, 0.9],
    omniglot_scripts: List[str] = None,
    resume_csv: str = None,
    prompt_version: str = "direct",
    prompt_tag: str = None,
):
    """
    Run the Scale Illusion experiment.
    Tests scale invariance by comparing full-size images with scaled versions (with padding).

    Args:
        alphabet_dir: Base directory containing times_new_roman
        omniglot_dir: Base directory containing omniglot
        model_config: Model configuration dictionary
        num_samples: (Unused) kept for backward compatibility
        positive_ratio: Ratio of positive pairs (0.5 = 50% positive, 50% negative)
        use_cot_prompt: Whether to use Chain-of-Thought prompt
        output_dir: Directory to save results
        log_path: Path to log file
        image_size: Size of images (default: 336 for Qwen-VL)
        rate_limit_delay: Delay between API calls (seconds)
        scale_factors: List of scale factors to test (0.0-1.0)
    """
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("SCALE ILLUSION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info("Total samples per dataset: all characters × all scales (no sampling)")
    logger.info(f"Positive ratio: {positive_ratio}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Scale factors: {scale_factors}")
    logger.info(f"Output directory: {output_dir}")

    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp(prefix="scale_illusion_")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        # Initialize model
        logger.info("Initializing model...")
        model_name = model_config.get("model_name", "unknown")
        model = InferenceModel(model_name)
        model_slug = _model_slug(model_name)
        logger.info(f"Model initialized: {model_name}")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise e

    # Load images
    logger.info("Loading alphabet images...")
    alphabet_images = load_alphabet_images(alphabet_dir)
    logger.info(f"Loaded {len(alphabet_images)} alphabet images")

    logger.info("Loading Omniglot images...")
    if omniglot_scripts:
        omniglot_scripts = [
            s for s in omniglot_scripts if str(s).strip().lower() != "english"
        ]
        logger.info(f"Limiting Omniglot scripts to: {omniglot_scripts}")
    else:
        logger.info("Using all Omniglot scripts under images_all")
    omniglot_images = load_omniglot_images(
        omniglot_dir, allowed_scripts=omniglot_scripts
    )
    logger.info(f"Loaded {len(omniglot_images)} Omniglot images")

    if len(alphabet_images) < 2:
        logger.error("Not enough alphabet images (need at least 2)")
        raise ValueError("Not enough alphabet images")

    if len(omniglot_images) < 2:
        logger.error("Not enough Omniglot images (need at least 2)")
        raise ValueError("Not enough Omniglot images")

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

    # Create experiment-specific directory (Gemini gets its own folder)
    if model_name == "gemini-2.5-pro":
        exp_dir_name = f"scale_illusion_{model_slug}"
    else:
        exp_dir_name = "scale_illusion"
    if resolved_tag:
        exp_dir_name = f"{exp_dir_name}_{resolved_tag}"
    exp_dir = os.path.join(output_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create sanity check subdirectory
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)

    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Sanity check directory: {sanity_dir}")

    # Create sanity check visualizations
    logger.info("Creating scale illusion sanity check visualizations...")
    try:
        sanity_path = os.path.join(
            sanity_dir, f"{model_slug}_scale_illusion_sanity_check.png"
        )
        create_scale_illusion_sanity_check(
            alphabet_dir=alphabet_dir,
            omniglot_dir=omniglot_dir,
            output_path=sanity_path,
            num_examples=6,
            image_size=image_size,
            scale_factors=scale_factors,
        )
        logger.info(f"Sanity check saved to: {sanity_path}")
    except Exception as e:
        logger.error(f"Failed to create sanity check: {e}")
        raise e

    is_llava = model_name in ("llava", "llava-1.5-7b") or str(model_name).startswith(
        "llava-hf/"
    )
    if is_llava:
        try:
            # Save a stitched example to show LLaVA's single-image presentation.
            a_char_id, a_img_path_first, a_img_path_second = random.choice(
                alphabet_images
            )
            tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp1.close()
            tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp2.close()
            img1 = (
                Image.open(a_img_path_first)
                .convert("RGB")
                .resize((image_size, image_size), Image.Resampling.LANCZOS)
            )
            img1.save(tmp1.name, "PNG")
            resize_with_padding(
                a_img_path_second,
                random.choice(scale_factors),
                tmp2.name,
                image_size=image_size,
            )
            stitched_path = os.path.join(
                sanity_dir, f"{model_slug}_llava_stitched_example.png"
            )
            stitch_two_images_with_labels(tmp1.name, tmp2.name, stitched_path)
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
            logger.info(f"LLaVA stitched sanity example saved to: {stitched_path}")
        except Exception as e:
            logger.warning(f"Failed to create LLaVA stitched sanity example: {e}")

    # Initialize results storage (optionally resume)
    results = []
    existing_keys = set()
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
                    row.get("script_name"),
                    row.get("char_id"),
                    scale_val,
                    _normalize_bool(row.get("is_positive")),
                )
                existing_keys.add(key)
            msg = (
                f"Resuming from {resume_csv} with {len(existing_keys)} existing entries"
            )
            logger.info(msg)
            print(msg, flush=True)
        except Exception as e:
            logger.warning(f"Failed to load resume CSV ({resume_csv}): {e}")

    csv_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion.csv")

    def append_result_row(row: Dict[str, any]) -> None:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # Run experiments for Alphabet dataset (English)
    logger.info("Running Alphabet experiments...")
    # Generate all permutations of (character, scale_factor)
    alphabet_permutations = [
        (char_id, img_path_first, img_path_second, scale)
        for char_id, img_path_first, img_path_second in alphabet_images
        for scale in scale_factors
    ]
    alphabet_samples = [
        (char_id, img_path_first, img_path_second, scale)
        for char_id, img_path_first, img_path_second, scale in alphabet_permutations
    ]
    logger.info(
        f"Using all {len(alphabet_samples)} Alphabet samples (all characters × all scales, both positive and negative)"
    )
    if existing_keys:
        expected_alpha = {
            ("alphabet", "English", char_id, float(scale_factor), is_positive)
            for char_id, _, _, scale_factor in alphabet_samples
            for is_positive in (True, False)
        }
        matched_alpha = sum(1 for k in expected_alpha if k in existing_keys)
        total_alpha = len(expected_alpha)
        remaining_alpha = max(total_alpha - matched_alpha, 0)
        msg = f"Alphabet remaining: {remaining_alpha}/{total_alpha} (matched {matched_alpha})"
        logger.info(msg)
        print(msg, flush=True)

    for idx, (char_id, img_path_first, img_path_second, scale_factor) in enumerate(
        tqdm(alphabet_samples, desc="Alphabet")
    ):
        for is_positive in (True, False):
            if (
                "alphabet",
                "English",
                char_id,
                scale_factor,
                is_positive,
            ) in existing_keys:
                continue

            # Image 1: Original (full size)
            img1 = Image.open(img_path_first).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
            temp_img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img1.close()
            img1.save(temp_img1.name, "PNG")
            img1_path = temp_img1.name

            if is_positive:
                # Positive pair: same character, scaled
                temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_img2.close()
                resize_with_padding(
                    img_path_second, scale_factor, temp_img2.name, image_size=image_size
                )
                img2_path = temp_img2.name
            else:
                # Negative pair: different character, scaled
                char_id2, img_path2_first, _ = random.choice(
                    [x for x in alphabet_images if x[0] != char_id]
                )
                temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_img2.close()
                resize_with_padding(
                    img_path2_first, scale_factor, temp_img2.name, image_size=image_size
                )
                img2_path = temp_img2.name

            # Run inference
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
                        "max_pixels": image_size * image_size * 2,  # Two images
                    }
                response = safe_infer(model, infer_payload)
                parsed = parse_response(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
            except Exception as e:
                logger.warning(f"Error in inference for Alphabet sample {idx}: {e}")
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    # Skip recording this row so we can rerun it later
                    os.unlink(temp_img1.name)
                    os.unlink(temp_img2.name)
                    if (
                        is_llava
                        and "stitched_tmp" in locals()
                        and os.path.exists(stitched_tmp.name)
                    ):
                        os.unlink(stitched_tmp.name)
                    continue
                # Skip recording any error responses
                os.unlink(temp_img1.name)
                os.unlink(temp_img2.name)
                if (
                    is_llava
                    and "stitched_tmp" in locals()
                    and os.path.exists(stitched_tmp.name)
                ):
                    os.unlink(stitched_tmp.name)
                continue

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            # Clean up temp files
            os.unlink(temp_img1.name)
            os.unlink(temp_img2.name)
            if (
                is_llava
                and "stitched_tmp" in locals()
                and os.path.exists(stitched_tmp.name)
            ):
                os.unlink(stitched_tmp.name)

            # Record results
            is_correct = (prediction == "yes" and is_positive) or (
                prediction == "no" and not is_positive
            )
            row = {
                "dataset": "alphabet",
                "script_name": "English",
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": scale_factor,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
            results.append(row)
            append_result_row(row)

    # Run experiments for Omniglot dataset
    logger.info("Running Omniglot experiments...")
    # Generate all permutations of (character, scale_factor) - use ALL characters at EVERY scale
    omniglot_permutations = [
        (script_name, char_id, img_path_first, img_path_second, scale)
        for script_name, char_id, img_path_first, img_path_second in omniglot_images
        for scale in scale_factors
    ]

    # Use ALL permutations for Omniglot (no sampling)
    omniglot_samples = [
        (script_name, char_id, img_path_first, img_path_second, scale)
        for script_name, char_id, img_path_first, img_path_second, scale in omniglot_permutations
    ]
    logger.info(
        f"Using all {len(omniglot_samples)} Omniglot samples (all characters × all scales, both positive and negative)"
    )
    if existing_keys:
        expected_omni = {
            ("omniglot", script_name, char_id, float(scale_factor), is_positive)
            for script_name, char_id, _, _, scale_factor in omniglot_samples
            for is_positive in (True, False)
        }
        matched_omni = sum(1 for k in expected_omni if k in existing_keys)
        total_omni = len(expected_omni)
        remaining_omni = max(total_omni - matched_omni, 0)
        msg = f"Omniglot remaining: {remaining_omni}/{total_omni} (matched {matched_omni})"
        logger.info(msg)
        print(msg, flush=True)

    for idx, (
        script_name,
        char_id,
        img_path_first,
        img_path_second,
        scale_factor,
    ) in enumerate(tqdm(omniglot_samples, desc="Omniglot")):
        for is_positive in (True, False):
            if (
                "omniglot",
                script_name,
                char_id,
                scale_factor,
                is_positive,
            ) in existing_keys:
                continue

            # Image 1: Original (full size)
            img1 = Image.open(img_path_first).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
            temp_img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img1.close()
            img1.save(temp_img1.name, "PNG")
            img1_path = temp_img1.name

            if is_positive:
                # Positive pair: same character, scaled
                temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_img2.close()
                resize_with_padding(
                    img_path_second, scale_factor, temp_img2.name, image_size=image_size
                )
                img2_path = temp_img2.name
            else:
                # Negative pair: different character from same script
                same_script = [
                    (s, c, p1, p2)
                    for s, c, p1, p2 in omniglot_images
                    if s == script_name and c != char_id
                ]
                if same_script:
                    script_name2, char_id2, img_path2_first, _ = random.choice(
                        same_script
                    )
                else:
                    script_name2, char_id2, img_path2_first, _ = random.choice(
                        [x for x in omniglot_images if x[1] != char_id]
                    )

                temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_img2.close()
                resize_with_padding(
                    img_path2_first, scale_factor, temp_img2.name, image_size=image_size
                )
                img2_path = temp_img2.name

            # Run inference
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
                        "max_pixels": image_size * image_size * 2,  # Two images
                    }
                response = safe_infer(model, infer_payload)
                parsed = parse_response(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
            except Exception as e:
                logger.warning(f"Error in inference for Omniglot sample {idx}: {e}")
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    os.unlink(temp_img1.name)
                    os.unlink(temp_img2.name)
                    if (
                        is_llava
                        and "stitched_tmp" in locals()
                        and os.path.exists(stitched_tmp.name)
                    ):
                        os.unlink(stitched_tmp.name)
                    continue
                os.unlink(temp_img1.name)
                os.unlink(temp_img2.name)
                if (
                    is_llava
                    and "stitched_tmp" in locals()
                    and os.path.exists(stitched_tmp.name)
                ):
                    os.unlink(stitched_tmp.name)
                continue

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)

            # Clean up temp files
            os.unlink(temp_img1.name)
            os.unlink(temp_img2.name)
            if (
                is_llava
                and "stitched_tmp" in locals()
                and os.path.exists(stitched_tmp.name)
            ):
                os.unlink(stitched_tmp.name)

            # Record results
            is_correct = (prediction == "yes" and is_positive) or (
                prediction == "no" and not is_positive
            )
            row = {
                "dataset": "omniglot",
                "script_name": script_name,
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": scale_factor,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
            results.append(row)
            append_result_row(row)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    # Convert to DataFrame from CSV (results are saved incrementally)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Analysis
    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)

    # Overall accuracy by dataset
    alphabet_df = df[df["dataset"] == "alphabet"]
    omniglot_df = df[df["dataset"] == "omniglot"]

    alphabet_acc = alphabet_df["is_correct"].mean() * 100
    omniglot_acc = omniglot_df["is_correct"].mean() * 100
    generalization_gap_omniglot = alphabet_acc - omniglot_acc

    logger.info(
        f"Alphabet Accuracy: {alphabet_acc:.2f}% ({alphabet_df['is_correct'].sum()}/{len(alphabet_df)})"
    )
    logger.info(
        f"Omniglot Accuracy: {omniglot_acc:.2f}% ({omniglot_df['is_correct'].sum()}/{len(omniglot_df)})"
    )
    logger.info(
        f"Generalization Gap (Δ) Alphabet-Omniglot: {generalization_gap_omniglot:.2f}%"
    )

    # Classification metrics by pair type
    for dataset_name, dataset_df in [
        ("Alphabet", alphabet_df),
        ("Omniglot", omniglot_df),
    ]:
        # Positive pairs: True Positive Rate (Recall/Sensitivity)
        pos_df = dataset_df[dataset_df["is_positive"] == True]
        tp = len(pos_df[pos_df["prediction"] == "yes"])
        fn = len(pos_df[pos_df["prediction"] == "no"])
        tpr = (tp / len(pos_df) * 100) if len(pos_df) > 0 else 0
        fnr = (fn / len(pos_df) * 100) if len(pos_df) > 0 else 0

        # Negative pairs: True Negative Rate (Specificity)
        neg_df = dataset_df[dataset_df["is_positive"] == False]
        tn = len(neg_df[neg_df["prediction"] == "no"])
        fp = len(neg_df[neg_df["prediction"] == "yes"])
        tnr = (tn / len(neg_df) * 100) if len(neg_df) > 0 else 0
        fpr = (fp / len(neg_df) * 100) if len(neg_df) > 0 else 0

        logger.info(
            f"{dataset_name} - Positive pairs (Recall/TPR): {tpr:.2f}% (FNR: {fnr:.2f}%)"
        )
        logger.info(
            f"{dataset_name} - Negative pairs (Specificity/TNR): {tnr:.2f}% (FPR: {fpr:.2f}%)"
        )
        logger.info(
            f"{dataset_name} - Overall accuracy: {dataset_df['is_correct'].mean() * 100:.2f}%"
        )

    # Accuracy / TPR / TNR by script (Omniglot scripts + English)
    logger.info("Accuracy / TPR / TNR by script:")
    script_summary_rows = []
    for script_name in sorted(df["script_name"].dropna().unique()):
        script_df = df[df["script_name"] == script_name]
        m = _acc_tpr_tnr_from_subset(script_df)
        logger.info(
            f"  {script_name}: acc={m['accuracy']:.2f}%  TPR={m['tpr']:.2f}%  TNR={m['tnr']:.2f}% "
            f"({m['n_correct']}/{m['n_samples']})"
        )
        script_summary_rows.append({"script_name": script_name, **m})

    script_summary_df = pd.DataFrame(script_summary_rows)
    script_summary_path = os.path.join(
        exp_dir, f"{model_slug}_scale_illusion_script_summary.csv"
    )
    script_summary_df.to_csv(script_summary_path, index=False)
    logger.info(f"Script summary saved to: {script_summary_path}")

    # Accuracy / TPR / TNR by scale factor per script
    logger.info("Accuracy / TPR / TNR by scale factor (per script):")
    scale_summary_rows = []
    for script_name in sorted(df["script_name"].dropna().unique()):
        script_df = df[df["script_name"] == script_name]
        for scale_factor in scale_factors:
            scale_df = script_df[script_df["scale_factor"] == scale_factor]
            m = _acc_tpr_tnr_from_subset(scale_df)
            logger.info(
                f"  {script_name} @ {scale_factor:.1f}: acc={m['accuracy']:.2f}%  "
                f"TPR={m['tpr']:.2f}%  TNR={m['tnr']:.2f}% ({m['n_correct']}/{m['n_samples']})"
            )
            scale_summary_rows.append(
                {
                    "script_name": script_name,
                    "scale_factor": scale_factor,
                    **m,
                }
            )

    scale_summary_df = pd.DataFrame(scale_summary_rows)
    scale_summary_path = os.path.join(
        exp_dir, f"{model_slug}_scale_illusion_scale_summary.csv"
    )
    scale_summary_df.to_csv(scale_summary_path, index=False)
    logger.info(f"Scale summary saved to: {scale_summary_path}")

    def _omniglot_groups(omniglot_df: pd.DataFrame):
        script_acc = (
            omniglot_df.groupby("script_name")["is_correct"]
            .mean()
            .sort_values(ascending=False)
        )
        scripts = list(script_acc.index)
        if len(scripts) < 6:
            logger.warning(
                "Fewer than 6 Omniglot scripts; grouping as a single Omniglot bar."
            )
            return [("Omniglot (all)", omniglot_df)]

        top = scripts[:2]
        bottom = scripts[-2:]
        remaining = scripts[2:-2]
        if len(remaining) >= 2:
            mid_start = max(0, (len(remaining) - 2) // 2)
            middle = remaining[mid_start : mid_start + 2]
        else:
            middle = remaining

        groups = [
            ("Omniglot Top 2", omniglot_df[omniglot_df["script_name"].isin(top)]),
        ]
        if middle:
            groups.append(
                ("Omniglot Mid 2", omniglot_df[omniglot_df["script_name"].isin(middle)])
            )
        groups.append(
            ("Omniglot Bottom 2", omniglot_df[omniglot_df["script_name"].isin(bottom)])
        )
        return groups

    omniglot_groups = _omniglot_groups(omniglot_df)
    group_labels = [name for name, _ in omniglot_groups]
    group_dfs = [dfg for _, dfg in omniglot_groups]

    # Create visualization
    fig_width = max(14, 0.6 * (1 + len(group_labels)))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 10))

    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    datasets = ["Alphabet\n(High Semantic)"] + group_labels
    group_accs = [
        dfg["is_correct"].mean() * 100 if len(dfg) > 0 else 0 for dfg in group_dfs
    ]
    group_ns = [len(dfg) for dfg in group_dfs]
    accuracies = [alphabet_acc] + group_accs
    n_samples = [len(alphabet_df)] + group_ns
    colors = ["green" if acc > 50 else "red" for acc in accuracies]
    bars = ax1.bar(
        datasets, accuracies, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 100)
    for bar, acc, n in zip(bars, accuracies, n_samples):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{acc:.1f}%\n(n={n})",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Generalization Gap
    ax2 = axes[0, 1]
    group_gaps = [
        alphabet_acc - (dfg["is_correct"].mean() * 100 if len(dfg) > 0 else 0)
        for dfg in group_dfs
    ]
    gaps = group_gaps
    gap_labels = [f"Alphabet - {name}" for name in group_labels]
    gap_colors = ["red" if g > 20 else "orange" if g > 10 else "green" for g in gaps]
    bars = ax2.barh(
        gap_labels, gaps, color=gap_colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax2.axvline(0, color="black", linestyle="-", linewidth=1)
    ax2.axvline(
        20, color="red", linestyle="--", linewidth=1, label="Critical Threshold (20%)"
    )
    ax2.set_xlabel("Accuracy Difference (%)", fontsize=12)
    for bar, gap in zip(bars, gaps):
        ax2.text(
            gap / 2 if gap != 0 else 0,
            bar.get_y() + bar.get_height() / 2,
            f"Δ = {gap:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=11,
            color="white" if abs(gap) > 10 else "black",
        )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. Classification metrics by pair type
    ax3 = axes[1, 0]

    def calc_metrics(df):
        pos_df = df[df["is_positive"] == True]
        neg_df = df[df["is_positive"] == False]
        tpr = (
            (len(pos_df[pos_df["prediction"] == "yes"]) / len(pos_df) * 100)
            if len(pos_df) > 0
            else 0
        )
        tnr = (
            (len(neg_df[neg_df["prediction"] == "no"]) / len(neg_df) * 100)
            if len(neg_df) > 0
            else 0
        )
        return tpr, tnr, len(pos_df), len(neg_df)

    alphabet_tpr, alphabet_tnr, alphabet_pos_n, alphabet_neg_n = calc_metrics(
        alphabet_df
    )
    group_metrics = [calc_metrics(dfg) for dfg in group_dfs]

    pair_types = ["Positive Pairs\n(Recall/TPR)", "Negative Pairs\n(Specificity/TNR)"]
    x = np.arange(len(pair_types))
    dataset_labels = ["Alphabet"] + group_labels
    dataset_metrics = [
        (alphabet_tpr, alphabet_tnr, alphabet_pos_n, alphabet_neg_n)
    ] + group_metrics
    colors = ["steelblue", "coral", "orange", "slateblue", "seagreen", "teal", "gray"]
    num_datasets = len(dataset_labels)
    width = min(0.8 / num_datasets, 0.2)
    offsets = np.linspace(
        -width * (num_datasets - 1) / 2, width * (num_datasets - 1) / 2, num_datasets
    )
    bars_by_dataset = []
    for i, (label, metrics) in enumerate(zip(dataset_labels, dataset_metrics)):
        tpr, tnr, _, _ = metrics
        bars = ax3.bar(
            x + offsets[i],
            [tpr, tnr],
            width,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.7,
            edgecolor="black",
        )
        bars_by_dataset.append(bars)
    ax3.set_ylabel("Rate (%)", fontsize=12)
    ax3.axhline(
        50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)"
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(pair_types)
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    for bars, metrics in zip(bars_by_dataset, dataset_metrics):
        _, _, pos_n, neg_n = metrics
        for bar, n in zip(bars, [pos_n, neg_n]):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%\n(n={n})",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    # 4. Accuracy by scale factor
    ax4 = axes[1, 1]
    scale_labels = [f"{s:.1f}" for s in scale_factors]
    x = np.arange(len(scale_labels))

    alphabet_scale_accs = [
        (
            alphabet_df[alphabet_df["scale_factor"] == s]["is_correct"].mean() * 100
            if len(alphabet_df[alphabet_df["scale_factor"] == s]) > 0
            else 0
        )
        for s in scale_factors
    ]
    alphabet_scale_n = [
        len(alphabet_df[alphabet_df["scale_factor"] == s]) for s in scale_factors
    ]
    group_scale_accs = [
        [
            (
                dfg[dfg["scale_factor"] == s]["is_correct"].mean() * 100
                if len(dfg[dfg["scale_factor"] == s]) > 0
                else 0
            )
            for s in scale_factors
        ]
        for dfg in group_dfs
    ]
    group_scale_n = [
        [len(dfg[dfg["scale_factor"] == s]) for s in scale_factors] for dfg in group_dfs
    ]

    dataset_labels = ["Alphabet"] + group_labels
    dataset_scale_accs = [alphabet_scale_accs] + group_scale_accs
    dataset_scale_ns = [alphabet_scale_n] + group_scale_n
    num_datasets = len(dataset_labels)
    width = min(0.8 / num_datasets, 0.2)
    offsets = np.linspace(
        -width * (num_datasets - 1) / 2, width * (num_datasets - 1) / 2, num_datasets
    )
    bars_by_dataset = []
    for i, (label, scale_accs) in enumerate(zip(dataset_labels, dataset_scale_accs)):
        bars = ax4.bar(
            x + offsets[i],
            scale_accs,
            width,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.7,
            edgecolor="black",
        )
        bars_by_dataset.append(bars)
    ax4.set_ylabel("Accuracy (%)", fontsize=12)
    ax4.set_xlabel("Scale Factor", fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(scale_labels)
    ax4.set_ylim(0, 100)
    ax4.axhline(
        50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    for bars, ns in zip(bars_by_dataset, dataset_scale_ns):
        for bar, n in zip(bars, ns):
            height = bar.get_height()
            if height > 0 or n > 0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f"{height:.1f}%\n(n={n})",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f"{model_slug}_scale_illusion_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {plot_path}")
    plt.close()

    # Create separate plot for Recall by scale factor
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Filter to positive pairs only for Recall calculation
    alphabet_pos_df = alphabet_df[alphabet_df["is_positive"] == True].copy()
    group_pos_dfs = [dfg[dfg["is_positive"] == True].copy() for dfg in group_dfs]

    # Calculate Recall (TPR) by scale factor for positive pairs
    alphabet_recall_by_scale = {}
    group_recall_by_scale = [dict() for _ in group_dfs]
    alphabet_recall_n = {}
    group_recall_n = [dict() for _ in group_dfs]

    for scale_factor in scale_factors:
        alphabet_scale_df = alphabet_pos_df[
            alphabet_pos_df["scale_factor"] == scale_factor
        ]
        group_scale_dfs = [
            gdf[gdf["scale_factor"] == scale_factor] for gdf in group_pos_dfs
        ]

        # Recall = TP / (TP + FN) = predicted 'yes' / all positive pairs
        alphabet_recall = (
            (
                len(alphabet_scale_df[alphabet_scale_df["prediction"] == "yes"])
                / len(alphabet_scale_df)
                * 100
            )
            if len(alphabet_scale_df) > 0
            else np.nan
        )
        alphabet_recall_by_scale[scale_factor] = alphabet_recall
        for idx, scale_df in enumerate(group_scale_dfs):
            recall = (
                (len(scale_df[scale_df["prediction"] == "yes"]) / len(scale_df) * 100)
                if len(scale_df) > 0
                else np.nan
            )
            group_recall_by_scale[idx][scale_factor] = recall
        alphabet_recall_n[scale_factor] = len(alphabet_scale_df)
        for idx, scale_df in enumerate(group_scale_dfs):
            group_recall_n[idx][scale_factor] = len(scale_df)

    # Plot Recall by scale factor
    scale_labels = [f"{s:.1f}" for s in scale_factors]
    x = np.arange(len(scale_labels))
    width = 0.35

    alphabet_recalls = [alphabet_recall_by_scale.get(s, np.nan) for s in scale_factors]
    group_recalls = [
        [group_recall_by_scale[idx].get(s, np.nan) for s in scale_factors]
        for idx in range(len(group_dfs))
    ]

    dataset_labels = ["Alphabet"] + group_labels
    dataset_recalls = [alphabet_recalls] + group_recalls
    num_datasets = len(dataset_labels)
    width = min(0.8 / num_datasets, 0.2)
    offsets = np.linspace(
        -width * (num_datasets - 1) / 2, width * (num_datasets - 1) / 2, num_datasets
    )
    bars_by_dataset = []
    for i, (label, recalls) in enumerate(zip(dataset_labels, dataset_recalls)):
        bars = ax.bar(
            x + offsets[i],
            [r if not np.isnan(r) else 0 for r in recalls],
            width,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.7,
            edgecolor="black",
        )
        bars_by_dataset.append(bars)

    ax.set_ylabel("Recall (TPR) (%)", fontsize=12)
    ax.set_xlabel("Scale Factor", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_ylim(0, 100)
    ax.axhline(
        50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add sample counts
    recall_ns = [alphabet_recall_n] + group_recall_n
    recall_lookup = [alphabet_recall_by_scale] + group_recall_by_scale
    for bars, n_map, val_map in zip(bars_by_dataset, recall_ns, recall_lookup):
        for bar, scale_factor in zip(bars, scale_factors):
            height = bar.get_height()
            n = n_map.get(scale_factor, 0)
            recall_val = val_map.get(scale_factor, np.nan)
            if not np.isnan(recall_val) and n > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f"{recall_val:.1f}%\n(n={n})",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    recall_plot_path = os.path.join(
        exp_dir, f"{model_slug}_scale_illusion_recall_by_scale.png"
    )
    plt.savefig(recall_plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Recall by scale factor plot saved to: {recall_plot_path}")
    plt.close()

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {exp_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Illusion Experiment")
    parser.add_argument(
        "--alphabet_dir",
        type=str,
        required=True,
        help="Base directory containing times_new_roman",
    )
    parser.add_argument(
        "--omniglot_dir",
        type=str,
        required=True,
        help="Base directory containing omniglot",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        choices=[
            "OpenGVLab/InternVL2_5-8B",
            "allenai/Molmo2-8B",
            "OpenGVLab/InternVideo2_5_Chat_8B",
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
        help="Model to use",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of samples per dataset"
    )
    parser.add_argument(
        "--positive_ratio",
        type=float,
        default=0.5,
        help="Ratio of positive pairs (default: 0.5)",
    )
    parser.add_argument(
        "--use_cot_prompt", action="store_true", help="Use Chain-of-Thought prompt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="Image size (default: 336 for Qwen-VL)",
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.0,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.9],
        help="Scale factors for experiment (default: 0.1 0.3 0.5 0.9)",
    )
    parser.add_argument(
        "--omniglot_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Optional: only these Omniglot script folder names (case-insensitive). "
        "If omitted, use all scripts under images_all. "
        "English is the times_new_roman alphabet dataset, not an Omniglot folder.",
    )
    parser.add_argument(
        "--resume_csv",
        type=str,
        default=None,
        help="Optional CSV to resume from (skip existing entries)",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="direct",
        choices=["direct", "cot", "v2", "v3"],
        help="Prompt version to use (default: direct)",
    )
    parser.add_argument(
        "--prompt_tag",
        type=str,
        default=None,
        help="Optional tag appended to output path (e.g., prompt_2xxxx_)",
    )

    args = parser.parse_args()

    model_config = {"model_name": args.model}

    run_scale_illusion_experiment(
        alphabet_dir=args.alphabet_dir,
        omniglot_dir=args.omniglot_dir,
        model_config=model_config,
        num_samples=args.num_samples,
        positive_ratio=args.positive_ratio,
        use_cot_prompt=args.use_cot_prompt,
        output_dir=args.output_dir,
        log_path=args.log_path,
        image_size=args.image_size,
        rate_limit_delay=args.rate_limit_delay,
        scale_factors=args.scale_factors,
        omniglot_scripts=args.omniglot_scripts if args.omniglot_scripts else None,
        resume_csv=args.resume_csv,
        prompt_version=args.prompt_version,
        prompt_tag=args.prompt_tag,
    )
