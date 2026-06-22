"""
Experiment: The "Spatial Illusion" (Geometric Reasoning vs. Semantic Familiarity)

Hypothesis: VLMs rely on semantic familiarity rather than geometric reasoning when matching
rotated images. They memorize canonical views of familiar objects (like Latin letters) but fail
to generalize to unfamiliar shapes (like Omniglot characters).

The Generalization Gap (Δ):
- Δ = Accuracy_Latin - Accuracy_Omniglot
- If Δ > 20%: Proof that the model is blind to shape and only bound to semantics
- If Δ ≈ 0: The model truly understands geometry (unlikely)

Protocol:
- Group A (High Semantic Prior): Standard Latin Alphabets (A, B, C...)
- Group B (Low Semantic Prior): Omniglot characters from rare alphabets
- Positive Pairs: Image X (0°) vs. Image X (Rotated θ)
- Negative Pairs: Image X (0°) vs. Image Y (Rotated θ) - Hard negatives from same script
- Rotation Angles: Arbitrary angles (30°, 45°, 135°, 210°) to force geometric interpolation
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
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.infer import InferenceModel
from google.genai.errors import ServerError

# --- CONFIGURATION ---
# Precomputed rotations live under zmeurer's omniglot python dir (see run_rotation_recog.sh).
DEFAULT_ROTATION_DATA_DIR = (
    None
)
ALPHABET_SCRIPT_FOLDER = "times_new_roman"
ALPHABET_DISPLAY_NAME = "English"
# Default filter when restricting to precomputed Angle_* folders (10–90 step 10).
ROTATION_ANGLES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Prompt options
# Original prompt: Simple and neutral, doesn't bias toward semantic or geometric reasoning
# This allows semantic shortcuts for alphabet but forces geometric reasoning for Omniglot
# Note: Using "same thing/content" instead of "same two images" to avoid literal file-identity interpretation

PROMPT_COT = """Look at the following two images.

Do these two images show the same underlying character? One could be a rotated version of the other.

Answer with just "YES" or "NO" after your reasoning."""

PROMPT_DIRECT = """If I rotate the first image, can I get the second image? Answer in curly brackets, e.g. {Yes} or {No}."""
# PROMPT_DIRECT = """You are given two images, each containing a single character on a white background. Decide whether they depict the same underlying character, allowing for rotation. Answer with exactly YES or NO."""

PROMPT_V2 = """Compare the two images and decide if they show the same character.
Ignore differences in rotation. Answer with exactly YES or NO."""
PROMPT_V3 = """Do the two images show the same visual character, even if one is rotated? Answer YES or NO."""


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

    # If heights differ, pad shorter image to match height
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

    x1, y1 = margin, margin
    x2, y2 = margin * 2 + w1, margin
    canvas.paste(img1, (x1, y1))
    canvas.paste(img2, (x2, y2))

    canvas.save(output_path, "PNG")
    return output_path


# Backward-compatible alias used by size_rotation experiment
stitch_two_images_side_by_side = stitch_two_images_with_labels


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


@dataclass(frozen=True)
class RotationRecogCharacter:
    """One character/stroke entry matching omniglot/rotation_recog.py."""

    dataset: str
    script_name: str
    char_id: str
    img_id: str
    original_image: str
    transformed_image_dir: str
    char_paths: Tuple[str, ...]


def resolve_rotation_data_dir(
    base_dir: str = "", explicit_dir: Optional[str] = None
) -> str:
    """Resolve python dir with images_original + images_rotated.

    Defaults to zmeurer's precomputed rotation tree (DEFAULT_ROTATION_DATA_DIR).
    Pass explicit_dir (or --rotation_data_dir) only to override.
    """
    del base_dir  # kept for backward-compatible call sites
    data_dir = explicit_dir or DEFAULT_ROTATION_DATA_DIR
    if not os.path.isdir(os.path.join(data_dir, "images_original")):
        raise FileNotFoundError(f"images_original not found under {data_dir}")
    if not os.path.isdir(os.path.join(data_dir, "images_rotated")):
        raise FileNotFoundError(f"images_rotated not found under {data_dir}")
    return data_dir


def _pick_random_negative_image(char_paths: List[str], original_image: str) -> str:
    """Same negative sampling as omniglot/rotation_recog.py."""
    random_image = original_image
    while random_image == original_image:
        random_char = random.choice(char_paths)
        choices = [f for f in os.listdir(random_char) if f.lower().endswith(".png")]
        random_image = os.path.join(random_char, random.choice(choices))
    return random_image


def _list_precomputed_angles(
    transformed_image_dir: str, rotation_angles: Optional[List[float]] = None
) -> List[int]:
    angles = sorted(
        int(name.replace("Angle_", ""))
        for name in os.listdir(transformed_image_dir)
        if name.startswith("Angle_")
    )
    if rotation_angles is not None:
        allowed = {int(a) for a in rotation_angles}
        angles = [a for a in angles if a in allowed]
    return angles


def _precomputed_rotated_image_path(transformed_image_dir: str, angle: float) -> str:
    angle_dir = os.path.join(transformed_image_dir, f"Angle_{int(angle)}")
    angle_file = os.listdir(angle_dir)[0]
    return os.path.join(angle_dir, angle_file)


def rotation_recog_image_pair(
    original_image: str,
    transformed_image_dir: str,
    angle: float,
    is_positive: bool,
    random_image: str,
) -> Tuple[str, str]:
    """
    Build image pair exactly like rotation_recog.py:
    positive -> [original, precomputed rotation]; negative -> [original, upright random].
    """
    if is_positive:
        second_image = _precomputed_rotated_image_path(transformed_image_dir, angle)
    else:
        second_image = random_image
    return original_image, second_image


def _load_rotation_recog_script_entries(
    data_dir: str,
    script_folder: str,
    *,
    dataset: str,
    display_script_name: str,
) -> List[RotationRecogCharacter]:
    original_folder = os.path.join(data_dir, "images_original", script_folder)
    rotated_folder = os.path.join(data_dir, "images_rotated", script_folder)
    if not os.path.isdir(original_folder) or not os.path.isdir(rotated_folder):
        return []

    char_paths = sorted(
        os.path.join(original_folder, name)
        for name in os.listdir(original_folder)
        if os.path.isdir(os.path.join(original_folder, name))
    )
    transformed_char_paths = sorted(
        os.path.join(rotated_folder, name)
        for name in os.listdir(rotated_folder)
        if os.path.isdir(os.path.join(rotated_folder, name))
    )

    entries: List[RotationRecogCharacter] = []
    for char_path, transformed_char_path in zip(char_paths, transformed_char_paths):
        char_id = os.path.basename(transformed_char_path)
        image_paths = sorted(
            os.path.join(char_path, name)
            for name in os.listdir(char_path)
            if name.lower().endswith(".png")
        )
        if not image_paths:
            continue
        image_paths = [image_paths[0]]

        transformed_image_paths = sorted(
            os.path.join(transformed_char_path, name)
            for name in os.listdir(transformed_char_path)
            if os.path.isdir(os.path.join(transformed_char_path, name))
        )
        if not transformed_image_paths:
            continue
        transformed_image_paths = [transformed_image_paths[0]]

        for original_image, transformed_image_dir in zip(
            image_paths, transformed_image_paths
        ):
            img_id = os.path.basename(transformed_image_dir)
            entries.append(
                RotationRecogCharacter(
                    dataset=dataset,
                    script_name=display_script_name,
                    char_id=char_id,
                    img_id=img_id,
                    original_image=original_image,
                    transformed_image_dir=transformed_image_dir,
                    char_paths=tuple(char_paths),
                )
            )
    return entries


def load_rotation_recog_alphabet(data_dir: str) -> List[RotationRecogCharacter]:
    return _load_rotation_recog_script_entries(
        data_dir,
        ALPHABET_SCRIPT_FOLDER,
        dataset="alphabet",
        display_script_name=ALPHABET_DISPLAY_NAME,
    )


def load_rotation_recog_omniglot(
    data_dir: str,
    allowed_scripts: List[str] = None,
    max_scripts: Optional[int] = None,
) -> List[RotationRecogCharacter]:
    original_root = os.path.join(data_dir, "images_original")
    allowed_set = {s.lower() for s in allowed_scripts} if allowed_scripts else None
    script_names = sorted(
        name
        for name in os.listdir(original_root)
        if os.path.isdir(os.path.join(original_root, name))
        and name != ALPHABET_SCRIPT_FOLDER
    )
    if allowed_set is not None:
        script_names = [s for s in script_names if s.lower() in allowed_set]
    if max_scripts is not None and max_scripts > 0:
        script_names = script_names[:max_scripts]

    entries: List[RotationRecogCharacter] = []
    for script_name in script_names:
        entries.extend(
            _load_rotation_recog_script_entries(
                data_dir,
                script_name,
                dataset="omniglot",
                display_script_name=script_name,
            )
        )
    return entries


def load_alphabet_images(
    alphabet_dir: str, rotation_data_dir: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Backward-compatible loader returning (char_id, original_image_path).
    Uses images_original/times_new_roman from the rotation data dir.
    """
    data_dir = resolve_rotation_data_dir(alphabet_dir, rotation_data_dir)
    return [
        (entry.char_id, entry.original_image)
        for entry in load_rotation_recog_alphabet(data_dir)
    ]


def load_omniglot_images(
    omniglot_dir: str,
    allowed_scripts: List[str] = None,
    max_scripts: Optional[int] = None,
    rotation_data_dir: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """
    Backward-compatible loader returning (script_name, char_id, original_image_path).
    Uses images_original/{script} from the rotation data dir (first stroke only).
    """
    data_dir = resolve_rotation_data_dir(omniglot_dir, rotation_data_dir)
    return [
        (entry.script_name, entry.char_id, entry.original_image)
        for entry in load_rotation_recog_omniglot(
            data_dir, allowed_scripts=allowed_scripts, max_scripts=max_scripts
        )
    ]


def build_rotation_experiment_jobs(
    alphabet_images=None,
    omniglot_images=None,
    rotation_angles: Optional[List[float]] = None,
    *,
    rotation_data_dir: Optional[str] = None,
    omniglot_dir: Optional[str] = None,
    allowed_scripts: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build exhaustive jobs matching rotation_recog schedule.
    alphabet_images / omniglot_images are ignored (kept for backward compatibility).
    """
    base_dir = omniglot_dir or DEFAULT_ROTATION_DATA_DIR
    data_dir = resolve_rotation_data_dir(
        base_dir, rotation_data_dir or DEFAULT_ROTATION_DATA_DIR
    )
    if rotation_angles is None:
        rotation_angles = ROTATION_ANGLES

    def _jobs_for_entries(
        entries: List[RotationRecogCharacter],
    ) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        for entry in entries:
            random_image = _pick_random_negative_image(
                list(entry.char_paths), entry.original_image
            )
            for angle in _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            ):
                for is_positive in (True, False):
                    jobs.append(
                        {
                            "dataset": entry.dataset,
                            "script_name": entry.script_name,
                            "char_id": entry.char_id,
                            "img_id": entry.img_id,
                            "original_image": entry.original_image,
                            "transformed_image_dir": entry.transformed_image_dir,
                            "angle": float(angle),
                            "is_positive": is_positive,
                            "random_image": random_image,
                        }
                    )
        return jobs

    alphabet_jobs = _jobs_for_entries(load_rotation_recog_alphabet(data_dir))
    omniglot_jobs = _jobs_for_entries(
        load_rotation_recog_omniglot(data_dir, allowed_scripts=allowed_scripts)
    )
    return alphabet_jobs, omniglot_jobs


def make_csv_row_appender(
    csv_path: str,
    fieldnames: Optional[List[str]] = None,
) -> Callable[[Dict[str, Any]], None]:
    """Append one result row to csv_path (writes header on first row)."""

    def append_result_row(row: Dict[str, Any]) -> None:
        nonlocal fieldnames
        if fieldnames is None:
            fieldnames = list(row.keys())
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({name: row.get(name) for name in fieldnames})

    return append_result_row


def process_rotation_jobs(
    jobs: List[Dict[str, Any]],
    *,
    model,
    raw_model_name: str,
    model_name: str,
    alphabet_images=None,
    omniglot_by_script=None,
    temp_dir: str = None,
    prompt_template: str,
    image_size: int,
    rate_limit_delay: float,
    logger,
    results: Dict[str, List],
    sanity_dir: str = None,
    saved_stitched_example: Dict[str, bool] = None,
    infer_and_parse: Optional[Callable] = None,
    extra_field_names: Optional[List[str]] = None,
    existing_keys: Optional[set] = None,
    append_result_row: Optional[Callable[[Dict[str, Any]], None]] = None,
    row_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Run rotation_recog-style jobs (precomputed rotations, shared negative per char)."""
    del alphabet_images, omniglot_by_script, temp_dir
    is_llava = raw_model_name in ("llava", "llava-1.5-7b") or str(
        raw_model_name
    ).startswith("llava-hf/")
    extra_field_names = extra_field_names or []

    pbar = tqdm(jobs, desc="rotation")
    for job in pbar:
        dataset = job["dataset"]
        pbar.set_description(str(dataset))
        script_name = job["script_name"]
        char_id = job["char_id"]
        angle = float(job["angle"])
        is_positive = bool(job["is_positive"])

        if (
            existing_keys
            and (
                dataset,
                script_name,
                char_id,
                angle,
                is_positive,
            )
            in existing_keys
        ):
            continue

        img1_path, img2_path = rotation_recog_image_pair(
            job["original_image"],
            job["transformed_image_dir"],
            angle,
            is_positive,
            job["random_image"],
        )

        max_pixels = infer_max_pixels_for_paths(
            [img1_path, img2_path], image_size * image_size * 2
        )

        text_prompt = (
            prompt_template(job) if callable(prompt_template) else prompt_template
        )

        stitched_tmp = None
        try:
            if is_llava:
                stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                stitched_tmp.close()
                stitch_two_images_with_labels(img1_path, img2_path, stitched_tmp.name)
                infer_payload = {
                    "image_path": stitched_tmp.name,
                    "text_prompt": text_prompt,
                }
            else:
                infer_payload = {
                    "image_paths": [img1_path, img2_path],
                    "text_prompt": text_prompt,
                    "max_pixels": max_pixels,
                }

            if infer_and_parse is not None:
                parsed = infer_and_parse(model, infer_payload, job)
                prediction = parsed["prediction"]
                response_clean = parsed.get(
                    "response_clean", parsed.get("response", "")
                )
                extra_values = {k: parsed[k] for k in extra_field_names if k in parsed}
            else:
                response = safe_infer(model, infer_payload)
                parsed = parse_response(response)
                prediction = parsed["answer"]
                response_clean = parsed["response_clean"]
                extra_values = {}
        except Exception as e:
            logger.warning(
                f"Error in inference for {dataset}/{script_name}/{char_id} "
                f"angle={angle} pos={is_positive}: {e}"
            )
            continue
        finally:
            if stitched_tmp is not None and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        ground_truth = "yes" if is_positive else "no"
        is_correct = (prediction == "yes" and is_positive) or (
            prediction == "no" and not is_positive
        )
        row = {
            "dataset": dataset,
            "script_name": script_name,
            "char_id": char_id,
            "pair_type": "positive" if is_positive else "negative",
            "angle": angle,
            "is_positive": is_positive,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": is_correct,
            "response": response_clean,
            "response_clean": response_clean,
            **extra_values,
        }
        if row_extra:
            row.update(row_extra)
        if "mcq_a_means_yes" in job:
            row["mcq_a_means_yes"] = job["mcq_a_means_yes"]
        for key, value in row.items():
            results.setdefault(key, []).append(value)
        if append_result_row is not None:
            append_result_row(row)

        if (
            saved_stitched_example is not None
            and sanity_dir
            and not saved_stitched_example.get(dataset, False)
        ):
            out_path = os.path.join(
                sanity_dir, f"{model_name}_{dataset}_rotation_recog_example.png"
            )
            stitch_two_images_with_labels(img1_path, img2_path, out_path)
            saved_stitched_example[dataset] = True


def _run_rotation_recog_dataset(
    entries: List[RotationRecogCharacter],
    *,
    model,
    raw_model_name: str,
    prompt_template: str,
    image_size: int,
    rate_limit_delay: float,
    rotation_angles: Optional[List[float]],
    logger,
    is_llava: bool,
    existing_keys: set,
    results: List[Dict[str, Any]],
    append_result_row: Callable[[Dict[str, Any]], None],
    desc: str,
) -> None:
    for entry in tqdm(entries, desc=desc):
        random_image = _pick_random_negative_image(
            list(entry.char_paths), entry.original_image
        )
        angles = _list_precomputed_angles(entry.transformed_image_dir, rotation_angles)
        for angle in angles:
            for is_positive in (True, False):
                if (
                    entry.dataset,
                    entry.script_name,
                    entry.char_id,
                    float(angle),
                    is_positive,
                ) in existing_keys:
                    continue

                img1_path, img2_path = rotation_recog_image_pair(
                    entry.original_image,
                    entry.transformed_image_dir,
                    angle,
                    is_positive,
                    random_image,
                )
                max_pixels = infer_max_pixels_for_paths(
                    [img1_path, img2_path], image_size * image_size * 2
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
                            "max_pixels": max_pixels,
                        }
                    response = safe_infer(model, infer_payload)
                    parsed = parse_response(response)
                    prediction = parsed["answer"]
                    response_clean = parsed["response_clean"]
                except Exception as e:
                    logger.warning(
                        f"Error in inference for {entry.dataset}/"
                        f"{entry.script_name}/{entry.char_id} angle={angle}: {e}"
                    )
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        continue
                    continue
                finally:
                    if (
                        is_llava
                        and stitched_tmp is not None
                        and os.path.exists(stitched_tmp.name)
                    ):
                        os.unlink(stitched_tmp.name)

                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)

                is_correct = (prediction == "yes" and is_positive) or (
                    prediction == "no" and not is_positive
                )
                row = {
                    "dataset": entry.dataset,
                    "script_name": entry.script_name,
                    "char_id": entry.char_id,
                    "is_positive": is_positive,
                    "angle": angle,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "response": response_clean,
                }
                results.append(row)
                append_result_row(row)


def _load_image_grayscale(image_path: str) -> np.ndarray:
    """Load as grayscale, matching omniglot/rotator.py."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is not None:
        return gray
    rgb = np.array(Image.open(image_path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rotate_image_smooth(
    image: np.ndarray,
    angle: float,
    resampling: int = cv2.INTER_LINEAR,
    ref_box: bool = False,
    reshape: bool = False,
    bg_value: int = 255,
) -> np.ndarray:
    """
    Same rotation as omniglot/rotator.py ExactRotator.rotate_image_smooth.
    reshape=False keeps native WxH; new pixels are white. No hard binarization.
    """
    image = image.copy()
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)

    if ref_box:
        cv2.rectangle(image, (1, 1), (width - 2, height - 2), (0,), thickness=1)

    rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), scale=1.0)
    if reshape:
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]
        width, height = new_width, new_height

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=resampling,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_value,
    )


def native_pair_canvas_size(image_paths: List[str]) -> Tuple[int, int]:
    """Max native width/height across pair images (padding only, never upscale)."""
    max_w, max_h = 0, 0
    for path in image_paths:
        gray = _load_image_grayscale(path)
        h, w = gray.shape[:2]
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h


def _center_gray_on_canvas(
    gray: np.ndarray, canvas_w: int, canvas_h: int, bg: int = 255
) -> np.ndarray:
    """Place grayscale image on a larger white canvas without resizing."""
    h, w = gray.shape[:2]
    if w > canvas_w or h > canvas_h:
        raise ValueError(
            f"Image {w}x{h} does not fit on canvas {canvas_w}x{canvas_h} without scaling"
        )
    canvas = np.full((canvas_h, canvas_w), bg, dtype=np.uint8)
    x0 = (canvas_w - w) // 2
    y0 = (canvas_h - h) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = gray
    return canvas


def _save_grayscale_png(gray: np.ndarray, output_path: str) -> str:
    cv2.imwrite(output_path, gray)
    return output_path


def prepare_reference_image(
    image_path: str,
    output_path: str,
    canvas_w: int = None,
    canvas_h: int = None,
) -> str:
    """0° reference at native resolution (same grayscale path as rotated image)."""
    gray = _load_image_grayscale(image_path)
    h, w = gray.shape[:2]
    if canvas_w is None or canvas_h is None:
        canvas_w, canvas_h = w, h
    if w == canvas_w and h == canvas_h:
        out = gray
    else:
        out = _center_gray_on_canvas(gray, canvas_w, canvas_h)
    return _save_grayscale_png(out, output_path)


def prepare_rotated_image(
    image_path: str,
    angle_deg: float,
    output_path: str,
    canvas_w: int = None,
    canvas_h: int = None,
    ref_box: bool = False,
) -> str:
    """Rotate with omniglot/rotator.py smooth path (same WxH, white padding)."""
    gray = _load_image_grayscale(image_path)
    if abs(float(angle_deg)) < 1e-6:
        rotated = gray.copy()
    else:
        rotated = rotate_image_smooth(
            gray,
            angle_deg,
            resampling=cv2.INTER_LINEAR,
            ref_box=ref_box,
            reshape=False,
        )
    h, w = rotated.shape[:2]
    if canvas_w is None or canvas_h is None:
        canvas_w, canvas_h = w, h
    if w == canvas_w and h == canvas_h:
        return _save_grayscale_png(rotated, output_path)
    return _save_grayscale_png(
        _center_gray_on_canvas(rotated, canvas_w, canvas_h), output_path
    )


def _load_image_rgb(image_path: str) -> Image.Image:
    """RGB load for matplotlib display only."""
    gray = _load_image_grayscale(image_path)
    return Image.fromarray(gray).convert("RGB")


def save_image_native(image_path: str, output_path: str) -> str:
    """Save image as RGB at native resolution (no resize, no crop)."""
    Image.open(image_path).convert("RGB").save(output_path, "PNG")
    return output_path


def _fit_display(img: Image.Image, max_side: int = 200) -> Image.Image:
    """Uniform downscale for matplotlib only; preserves aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
    )


def infer_max_pixels_for_paths(image_paths: List[str], fallback: int) -> int:
    total = 0
    for path in image_paths:
        with Image.open(path) as im:
            w, h = im.size
        total += w * h
    return max(total, fallback)


def rotate_image(
    image_path: str,
    angle: float,
    output_path: str = None,
    image_size: int = None,
    canvas_w: int = None,
    canvas_h: int = None,
) -> str:
    """Rotate at scale 1.0 (optionally on a fixed pair canvas). Used by size_rotation paths."""
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"rotated_{random.randint(0, 999999)}.png")
    return prepare_rotated_image(
        image_path, angle, output_path, canvas_w=canvas_w, canvas_h=canvas_h
    )


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


def create_spatial_illusion_sanity_check(
    alphabet_dir: str,
    omniglot_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
    rotation_angles: List[float] = None,
    rotation_data_dir: Optional[str] = None,
):
    """
    Create sanity check visualization for rotation experiment.
    Uses precomputed images_rotated pairs (rotation_recog protocol).
    """
    if rotation_angles is None:
        rotation_angles = ROTATION_ANGLES

    data_dir = resolve_rotation_data_dir(omniglot_dir, rotation_data_dir)
    alphabet_entries = load_rotation_recog_alphabet(data_dir)
    omniglot_entries = load_rotation_recog_omniglot(data_dir)

    if len(alphabet_entries) < 1 or len(omniglot_entries) < 1:
        raise ValueError("Not enough images for sanity check")

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 2.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        if i < num_examples // 2:
            entry = random.choice(alphabet_entries)
            is_positive = i % 2 == 0
            angles = _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            )
            if not angles:
                continue
            angle = random.choice(angles)
            random_image = _pick_random_negative_image(
                list(entry.char_paths), entry.original_image
            )
            img1_path, img2_path = rotation_recog_image_pair(
                entry.original_image,
                entry.transformed_image_dir,
                angle,
                is_positive,
                random_image,
            )
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            if is_positive:
                label = (
                    f"Alphabet: Positive\n{entry.char_id} "
                    f"(same stroke, {int(angle)}°)"
                )
            else:
                label = (
                    f"Alphabet: Negative\n{entry.char_id} vs random "
                    f"({int(angle)}°, distractor upright)"
                )
        else:
            entry = random.choice(omniglot_entries)
            is_positive = i % 2 == 0
            angles = _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            )
            if not angles:
                continue
            angle = random.choice(angles)
            random_image = _pick_random_negative_image(
                list(entry.char_paths), entry.original_image
            )
            img1_path, img2_path = rotation_recog_image_pair(
                entry.original_image,
                entry.transformed_image_dir,
                angle,
                is_positive,
                random_image,
            )
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            if is_positive:
                label = (
                    f"Omniglot: Positive\n{entry.script_name}/{entry.char_id} "
                    f"(same stroke, {int(angle)}°)"
                )
            else:
                label = (
                    f"Omniglot: Negative\n{entry.script_name}/{entry.char_id} vs "
                    f"random ({int(angle)}°, distractor upright)"
                )

        axes[i, 0].imshow(_fit_display(img1))
        axes[i, 0].axis("off")
        axes[i, 0].set_aspect("equal")
        axes[i, 1].imshow(_fit_display(img2))
        axes[i, 1].axis("off")
        axes[i, 1].set_aspect("equal")

        if i == 0:
            axes[i, 0].text(
                0.5,
                -0.1,
                "Original\n(0°)",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            axes[i, 1].text(
                0.5,
                -0.1,
                "Second image\n(rotated or upright distractor)",
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
    print(f"Spatial illusion sanity check visualization saved to: {output_path}")


def create_size_rotation_sanity_check(
    alphabet_dir: str,
    omniglot_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
    scale_factors: List[float] = [0.5, 0.7, 0.9],
):
    """
    Create sanity check visualization for size + rotation experiment.
    Shows 3 columns: original, smaller (with padding), rotated smaller.
    """
    alphabet_images = load_alphabet_images(alphabet_dir)
    omniglot_images = load_omniglot_images(omniglot_dir, max_scripts=5)

    if len(alphabet_images) < 2 or len(omniglot_images) < 2:
        raise ValueError("Not enough images for sanity check")

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 2 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        scale_factor = random.choice(scale_factors)
        angle = random.choice(ROTATION_ANGLES)

        if i < num_examples // 2:
            # Alphabet examples
            char_id, img_path = random.choice(alphabet_images)
            is_positive = i % 2 == 0

            # Original image
            img1 = Image.open(img_path).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

            # Smaller image (with padding)
            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path, scale_factor, temp_small.name, image_size=image_size
            )
            img_small = Image.open(temp_small.name).convert("RGB")
            os.unlink(temp_small.name)

            if is_positive:
                # Positive pair: same stroke, smaller, then rotated
                temp_small2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_small2.close()
                resize_with_padding(
                    img_path,
                    scale_factor,
                    temp_small2.name,
                    image_size=image_size,
                )
                temp_rotated = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_rotated.close()
                rotate_image(
                    temp_small2.name, angle, temp_rotated.name, image_size=image_size
                )
                img_rotated = Image.open(temp_rotated.name).convert("RGB")
                os.unlink(temp_rotated.name)
                os.unlink(temp_small2.name)
                label = f"Alphabet: Positive\n{char_id} (scale={scale_factor:.1f}, {angle}°)"
            else:
                # Negative pair: different character
                char_id2, img_path2 = random.choice(
                    [x for x in alphabet_images if x[0] != char_id]
                )
                temp_small2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_small2.close()
                resize_with_padding(
                    img_path2,
                    scale_factor,
                    temp_small2.name,
                    image_size=image_size,
                )
                temp_rotated = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_rotated.close()
                rotate_image(
                    temp_small2.name, angle, temp_rotated.name, image_size=image_size
                )
                img_rotated = Image.open(temp_rotated.name).convert("RGB")
                os.unlink(temp_rotated.name)
                os.unlink(temp_small2.name)
                label = f"Alphabet: Negative\n{char_id} vs {char_id2} (scale={scale_factor:.1f}, {angle}°)"

            dataset = "Alphabet"
        else:
            # Omniglot examples
            script_name, char_id, img_path = random.choice(omniglot_images)
            is_positive = i % 2 == 0

            # Original image
            img1 = Image.open(img_path).convert("RGB")
            if img1.size != (image_size, image_size):
                img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

            # Smaller image (with padding)
            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path, scale_factor, temp_small.name, image_size=image_size
            )
            img_small = Image.open(temp_small.name).convert("RGB")
            os.unlink(temp_small.name)

            if is_positive:
                # Positive pair: same first stroke, smaller, then rotated
                temp_small2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_small2.close()
                resize_with_padding(
                    img_path,
                    scale_factor,
                    temp_small2.name,
                    image_size=image_size,
                )
                temp_rotated = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_rotated.close()
                rotate_image(
                    temp_small2.name, angle, temp_rotated.name, image_size=image_size
                )
                img_rotated = Image.open(temp_rotated.name).convert("RGB")
                os.unlink(temp_rotated.name)
                os.unlink(temp_small2.name)
                label = f"Omniglot: Positive\n{script_name}/{char_id} (scale={scale_factor:.1f}, {angle}°)"
            else:
                same_script = [
                    (s, c, p)
                    for s, c, p in omniglot_images
                    if s == script_name and c != char_id
                ]
                if same_script:
                    script_name2, char_id2, img_path2 = random.choice(same_script)
                else:
                    script_name2, char_id2, img_path2 = random.choice(
                        [x for x in omniglot_images if x[1] != char_id]
                    )
                temp_small2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_small2.close()
                resize_with_padding(
                    img_path2,
                    scale_factor,
                    temp_small2.name,
                    image_size=image_size,
                )
                temp_rotated = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_rotated.close()
                rotate_image(
                    temp_small2.name, angle, temp_rotated.name, image_size=image_size
                )
                img_rotated = Image.open(temp_rotated.name).convert("RGB")
                os.unlink(temp_rotated.name)
                os.unlink(temp_small2.name)
                label = f"Omniglot: Negative\n{script_name}/{char_id} vs {script_name2}/{char_id2} (scale={scale_factor:.1f}, {angle}°)"

            dataset = "Omniglot"

        # Column 1: Original
        axes[i, 0].imshow(_fit_display(img1))
        axes[i, 0].axis("off")
        axes[i, 0].set_aspect("equal")

        # Column 2: Smaller (with padding)
        axes[i, 1].imshow(_fit_display(img_small))
        axes[i, 1].axis("off")
        axes[i, 1].set_aspect("equal")

        # Column 3: Rotated smaller
        axes[i, 2].imshow(_fit_display(img_rotated))
        axes[i, 2].axis("off")
        axes[i, 2].set_aspect("equal")

        # Add label below the row
        if i == 0:
            axes[i, 0].text(
                0.5,
                -0.1,
                "Original",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            axes[i, 1].text(
                0.5,
                -0.1,
                "Smaller",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            axes[i, 2].text(
                0.5,
                -0.1,
                "Rotated",
                transform=axes[i, 2].transAxes,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Add dataset and pair type label on the right
        axes[i, 2].text(
            1.1,
            0.5,
            label,
            transform=axes[i, 2].transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Size + rotation sanity check visualization saved to: {output_path}")


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


def run_spatial_illusion_experiment(
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
    rotation_angles: List[float] = None,
    omniglot_scripts: List[str] = None,
    resume_csv: str = None,
    prompt_version: str = "direct",
    prompt_tag: str = None,
    rotation_data_dir: str = None,
):
    """
    Run the Spatial Illusion rotation experiment using the rotation_recog protocol:
    precomputed images_rotated pairs, negative distractors upright (not rotated).

    Args:
        alphabet_dir: (Legacy) unused for image loading; kept for CLI compatibility
        omniglot_dir: (Legacy) unused for image loading; kept for CLI compatibility
        rotation_data_dir: zmeurer python dir with images_original/images_rotated
        model_config: Model configuration dictionary
        num_samples: (Unused) kept for backward compatibility
        positive_ratio: (Unused) both pos/neg run per char×angle
        use_cot_prompt: Whether to use Chain-of-Thought prompt
        output_dir: Directory to save results
        log_path: Path to log file
        image_size: Size of images (default: 336 for Qwen-VL)
        rate_limit_delay: Delay between API calls (seconds)
        rotation_angles: Filter precomputed Angle_* folders to these degrees
    """
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("SPATIAL ILLUSION (ROTATION) EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info(
        "Protocol: rotation_recog (images_original + precomputed images_rotated)"
    )
    logger.info("Total samples per dataset: all characters × all angles (no sampling)")
    logger.info(f"Positive ratio: {positive_ratio}")
    logger.info(f"Image size: {image_size}x{image_size}")
    if rotation_angles is None:
        rotation_angles = ROTATION_ANGLES
    logger.info(f"Rotation angles (filter): {rotation_angles}")
    logger.info(f"Output directory: {output_dir}")

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

    data_dir = resolve_rotation_data_dir(omniglot_dir, rotation_data_dir)
    logger.info(f"Rotation data dir (zmeurer): {data_dir}")

    logger.info("Loading alphabet entries (images_original/times_new_roman)...")
    alphabet_entries = load_rotation_recog_alphabet(data_dir)
    logger.info(f"Loaded {len(alphabet_entries)} alphabet entries")

    logger.info("Loading Omniglot entries (images_original/{script})...")
    if omniglot_scripts:
        omniglot_scripts = [
            s for s in omniglot_scripts if str(s).strip().lower() != "english"
        ]
        logger.info(f"Limiting Omniglot scripts to: {omniglot_scripts}")
    else:
        logger.info("Using all Omniglot scripts under images_original")
    omniglot_entries = load_rotation_recog_omniglot(
        data_dir, allowed_scripts=omniglot_scripts
    )
    logger.info(f"Loaded {len(omniglot_entries)} Omniglot entries")

    if len(alphabet_entries) < 2:
        logger.error("Not enough alphabet entries (need at least 2)")
        raise ValueError("Not enough alphabet entries")

    if len(omniglot_entries) < 2:
        logger.error("Not enough Omniglot entries (need at least 2)")
        raise ValueError("Not enough Omniglot entries")

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
        exp_dir_name = f"spatial_illusion_{model_slug}"
    else:
        exp_dir_name = "spatial_illusion"
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
            sanity_dir, f"{model_slug}_spatial_illusion_sanity_check.png"
        )
        create_spatial_illusion_sanity_check(
            alphabet_dir=alphabet_dir,
            omniglot_dir=omniglot_dir,
            output_path=sanity_path,
            num_examples=6,
            image_size=image_size,
            rotation_angles=rotation_angles,
            rotation_data_dir=data_dir,
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
            entry = random.choice(alphabet_entries)
            angles = _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            )
            llava_angle = random.choice(angles)
            img1_path, img2_path = rotation_recog_image_pair(
                entry.original_image,
                entry.transformed_image_dir,
                llava_angle,
                True,
                entry.original_image,
            )
            stitched_path = os.path.join(
                sanity_dir, f"{model_slug}_llava_stitched_example.png"
            )
            stitch_two_images_with_labels(img1_path, img2_path, stitched_path)
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
                scale_val = row.get("angle")
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

    csv_path = os.path.join(exp_dir, f"{model_slug}_spatial_illusion.csv")
    append_result_row = make_csv_row_appender(csv_path)

    # Run experiments (rotation_recog schedule: one negative per character)
    logger.info("Running Alphabet experiments...")
    if existing_keys:
        expected_alpha = {
            ("alphabet", entry.script_name, entry.char_id, float(angle), is_positive)
            for entry in alphabet_entries
            for angle in _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            )
            for is_positive in (True, False)
        }
        matched_alpha = sum(1 for k in expected_alpha if k in existing_keys)
        total_alpha = len(expected_alpha)
        remaining_alpha = max(total_alpha - matched_alpha, 0)
        msg = f"Alphabet remaining: {remaining_alpha}/{total_alpha} (matched {matched_alpha})"
        logger.info(msg)
        print(msg, flush=True)

    _run_rotation_recog_dataset(
        alphabet_entries,
        model=model,
        raw_model_name=model_name,
        prompt_template=prompt_template,
        image_size=image_size,
        rate_limit_delay=rate_limit_delay,
        rotation_angles=rotation_angles,
        logger=logger,
        is_llava=is_llava,
        existing_keys=existing_keys,
        results=results,
        append_result_row=append_result_row,
        desc="Alphabet",
    )

    logger.info("Running Omniglot experiments...")
    if existing_keys:
        expected_omni = {
            ("omniglot", entry.script_name, entry.char_id, float(angle), is_positive)
            for entry in omniglot_entries
            for angle in _list_precomputed_angles(
                entry.transformed_image_dir, rotation_angles
            )
            for is_positive in (True, False)
        }
        matched_omni = sum(1 for k in expected_omni if k in existing_keys)
        total_omni = len(expected_omni)
        remaining_omni = max(total_omni - matched_omni, 0)
        msg = f"Omniglot remaining: {remaining_omni}/{total_omni} (matched {matched_omni})"
        logger.info(msg)
        print(msg, flush=True)

    _run_rotation_recog_dataset(
        omniglot_entries,
        model=model,
        raw_model_name=model_name,
        prompt_template=prompt_template,
        image_size=image_size,
        rate_limit_delay=rate_limit_delay,
        rotation_angles=rotation_angles,
        logger=logger,
        is_llava=is_llava,
        existing_keys=existing_keys,
        results=results,
        append_result_row=append_result_row,
        desc="Omniglot",
    )

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
        exp_dir, f"{model_slug}_spatial_illusion_script_summary.csv"
    )
    script_summary_df.to_csv(script_summary_path, index=False)
    logger.info(f"Script summary saved to: {script_summary_path}")

    # Accuracy / TPR / TNR by rotation angle per script
    logger.info("Accuracy / TPR / TNR by rotation angle (per script):")
    angle_summary_rows = []
    for script_name in sorted(df["script_name"].dropna().unique()):
        script_df = df[df["script_name"] == script_name]
        for angle in rotation_angles:
            angle_df = script_df[script_df["angle"] == angle]
            m = _acc_tpr_tnr_from_subset(angle_df)
            logger.info(
                f"  {script_name} @ {int(angle)}°: acc={m['accuracy']:.2f}%  "
                f"TPR={m['tpr']:.2f}%  TNR={m['tnr']:.2f}% ({m['n_correct']}/{m['n_samples']})"
            )
            angle_summary_rows.append(
                {
                    "script_name": script_name,
                    "angle": angle,
                    **m,
                }
            )

    angle_summary_df = pd.DataFrame(angle_summary_rows)
    angle_summary_path = os.path.join(
        exp_dir, f"{model_slug}_spatial_illusion_angle_summary.csv"
    )
    angle_summary_df.to_csv(angle_summary_path, index=False)
    logger.info(f"Angle summary saved to: {angle_summary_path}")

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
    angle_labels = [f"{int(s)}°" for s in rotation_angles]
    x = np.arange(len(angle_labels))

    alphabet_scale_accs = [
        (
            alphabet_df[alphabet_df["angle"] == s]["is_correct"].mean() * 100
            if len(alphabet_df[alphabet_df["angle"] == s]) > 0
            else 0
        )
        for s in rotation_angles
    ]
    alphabet_scale_n = [
        len(alphabet_df[alphabet_df["angle"] == s]) for s in rotation_angles
    ]
    group_scale_accs = [
        [
            (
                dfg[dfg["angle"] == s]["is_correct"].mean() * 100
                if len(dfg[dfg["angle"] == s]) > 0
                else 0
            )
            for s in rotation_angles
        ]
        for dfg in group_dfs
    ]
    group_scale_n = [
        [len(dfg[dfg["angle"] == s]) for s in rotation_angles] for dfg in group_dfs
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
    ax4.set_xlabel("Rotation Angle", fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(angle_labels)
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
    plot_path = os.path.join(exp_dir, f"{model_slug}_spatial_illusion_plot.png")
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

    for angle in rotation_angles:
        alphabet_scale_df = alphabet_pos_df[alphabet_pos_df["angle"] == angle]
        group_scale_dfs = [gdf[gdf["angle"] == angle] for gdf in group_pos_dfs]

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
        alphabet_recall_by_scale[angle] = alphabet_recall
        for idx, scale_df in enumerate(group_scale_dfs):
            recall = (
                (len(scale_df[scale_df["prediction"] == "yes"]) / len(scale_df) * 100)
                if len(scale_df) > 0
                else np.nan
            )
            group_recall_by_scale[idx][angle] = recall
        alphabet_recall_n[angle] = len(alphabet_scale_df)
        for idx, scale_df in enumerate(group_scale_dfs):
            group_recall_n[idx][angle] = len(scale_df)

    # Plot Recall by scale factor
    angle_labels = [f"{int(s)}°" for s in rotation_angles]
    x = np.arange(len(angle_labels))
    width = 0.35

    alphabet_recalls = [
        alphabet_recall_by_scale.get(s, np.nan) for s in rotation_angles
    ]
    group_recalls = [
        [group_recall_by_scale[idx].get(s, np.nan) for s in rotation_angles]
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
    ax.set_xlabel("Rotation Angle", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(angle_labels)
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
        for bar, angle in zip(bars, rotation_angles):
            height = bar.get_height()
            n = n_map.get(angle, 0)
            recall_val = val_map.get(angle, np.nan)
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
        exp_dir, f"{model_slug}_spatial_illusion_recall_by_scale.png"
    )
    plt.savefig(recall_plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Recall by scale factor plot saved to: {recall_plot_path}")
    plt.close()

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {exp_dir}")

    return df


def run_size_rotation_experiment(
    alphabet_dir: str,
    omniglot_dir: str,
    model_config: Dict[str, str] = {"model_name": "qwen2.5-vl"},
    num_samples: int = 50,
    positive_ratio: float = 0.5,
    use_cot_prompt: bool = False,
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    scale_factors: List[float] = [0.5, 0.7, 0.9],
):
    """
    Run the Size + Rotation experiment.
    Tests rotation at different character sizes (with padding).

    Args:
        alphabet_dir: Base directory containing times_new_roman
        omniglot_dir: Base directory containing omniglot
        model_config: Model configuration dictionary
        num_samples: Number of test samples per dataset
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
    logger.info("SIZE + ROTATION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info(f"Samples per dataset: {num_samples}")
    logger.info(f"Positive ratio: {positive_ratio}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Scale factors: {scale_factors}")
    logger.info(f"Output directory: {output_dir}")

    # Create experiment-specific directory
    exp_dir = os.path.join(output_dir, "spatial_illusion_size_rotation")
    os.makedirs(exp_dir, exist_ok=True)

    # Create sanity check subdirectory
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)

    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Sanity check directory: {sanity_dir}")

    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp(prefix="size_rotation_")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        # Initialize model
        logger.info("Initializing model...")
        raw_model_name = model_config.get("model_name", "")
        model = InferenceModel(raw_model_name)
        if "qwen" in raw_model_name.lower():
            model_name = "qwen2.5_vl"
        elif "gemini" in raw_model_name.lower():
            model_name = "gemini2.5_pro"
        elif "llava" in raw_model_name.lower():
            model_name = "llava"
        else:
            model_name = (
                re.sub(r"[^a-z0-9]+", "_", raw_model_name.lower()).strip("_")
                or "unknown"
            )
        logger.info(f"Model initialized: {raw_model_name} (slug={model_name})")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise e

    # Create sanity check visualizations
    logger.info("Creating size + rotation sanity check visualizations...")
    try:
        sanity_path = os.path.join(
            sanity_dir, f"{model_name}_size_rotation_sanity_check.png"
        )
        create_size_rotation_sanity_check(
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

    # Load images
    logger.info("Loading alphabet images...")
    alphabet_images = load_alphabet_images(alphabet_dir)
    logger.info(f"Loaded {len(alphabet_images)} alphabet images")

    logger.info("Loading Omniglot images...")
    omniglot_images = load_omniglot_images(omniglot_dir, max_scripts=5)
    logger.info(f"Loaded {len(omniglot_images)} Omniglot images")

    if len(alphabet_images) < 2:
        logger.error("Not enough alphabet images (need at least 2)")
        raise ValueError("Not enough alphabet images")

    if len(omniglot_images) < 2:
        logger.error("Not enough Omniglot images (need at least 2)")
        raise ValueError("Not enough Omniglot images")

    # Select prompt
    prompt_template = PROMPT_COT if use_cot_prompt else PROMPT_DIRECT
    is_llava = raw_model_name in ("llava", "llava-1.5-7b", "llava-1.5-13b") or str(
        raw_model_name
    ).startswith("llava-hf/")

    # Initialize results storage
    results = []

    # Run experiments for Alphabet dataset
    logger.info("Running Alphabet experiments...")
    alphabet_samples = random.sample(
        alphabet_images, min(num_samples, len(alphabet_images))
    )

    for idx, (char_id, img_path) in enumerate(tqdm(alphabet_samples, desc="Alphabet")):
        # Randomly select scale factor and angle
        scale_factor = random.choice(scale_factors)
        angle = random.choice(ROTATION_ANGLES)
        is_positive = random.random() < positive_ratio

        # Image 1: original (0°), same canvas size as scale/rotation experiments
        img1 = Image.open(img_path).convert("RGB")
        if img1.size != (image_size, image_size):
            img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
        img1_path = os.path.join(temp_dir, f"alphabet_{idx}_img1.png")
        img1.save(img1_path, "PNG")

        # Image 2: scaled (with padding) then rotated
        if is_positive:
            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path, scale_factor, temp_small.name, image_size=image_size
            )
            img2_path = rotate_image(
                temp_small.name,
                angle,
                os.path.join(temp_dir, f"alphabet_{idx}_img2.png"),
                image_size=image_size,
            )
            os.unlink(temp_small.name)
        else:
            char_id2, img_path2 = random.choice(
                [x for x in alphabet_images if x[0] != char_id]
            )
            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path2, scale_factor, temp_small.name, image_size=image_size
            )
            img2_path = rotate_image(
                temp_small.name,
                angle,
                os.path.join(temp_dir, f"alphabet_{idx}_img2.png"),
                image_size=image_size,
            )
            os.unlink(temp_small.name)

        # Run inference
        stitched_tmp = None
        try:
            if is_llava:
                stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                stitched_tmp.close()
                stitch_two_images_side_by_side(img1_path, img2_path, stitched_tmp.name)
                prompt_for_model = (
                    "You are given a single composite image with two panels placed side by side.\n\n"
                    + prompt_template
                )
                infer_payload = {
                    "image_path": stitched_tmp.name,
                    "text_prompt": prompt_for_model,
                }
            else:
                infer_payload = {
                    "image_paths": [img1_path, img2_path],
                    "text_prompt": prompt_template,
                    "max_pixels": image_size * image_size * 2,  # Two images
                }
            response = safe_infer(model, infer_payload)
        except Exception as e:
            logger.warning(f"Error during model call for Alphabet sample {idx}: {e}")
            prediction = "unknown"
            response_clean = str(e)
        else:
            parsed = parse_response(response)
            prediction = parsed["answer"]
            response_clean = parsed["response_clean"]
        finally:
            if stitched_tmp is not None and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Record results
        is_correct = (prediction == "yes" and is_positive) or (
            prediction == "no" and not is_positive
        )
        results.append(
            {
                "dataset": "alphabet",
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": scale_factor,
                "angle": angle,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
        )

    # Run experiments for Omniglot dataset
    logger.info("Running Omniglot experiments...")
    omniglot_samples = random.sample(
        omniglot_images, min(num_samples, len(omniglot_images))
    )

    for idx, (script_name, char_id, img_path) in enumerate(
        tqdm(omniglot_samples, desc="Omniglot")
    ):
        scale_factor = random.choice(scale_factors)
        angle = random.choice(ROTATION_ANGLES)
        is_positive = random.random() < positive_ratio

        img1 = Image.open(img_path).convert("RGB")
        if img1.size != (image_size, image_size):
            img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
        img1_path = os.path.join(temp_dir, f"omniglot_{idx}_img1.png")
        img1.save(img1_path, "PNG")

        if is_positive:
            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path, scale_factor, temp_small.name, image_size=image_size
            )
            img2_path = rotate_image(
                temp_small.name,
                angle,
                os.path.join(temp_dir, f"omniglot_{idx}_img2.png"),
                image_size=image_size,
            )
            os.unlink(temp_small.name)
        else:
            same_script = [
                (s, c, p)
                for s, c, p in omniglot_images
                if s == script_name and c != char_id
            ]
            if same_script:
                script_name2, char_id2, img_path2 = random.choice(same_script)
            else:
                script_name2, char_id2, img_path2 = random.choice(
                    [x for x in omniglot_images if x[1] != char_id]
                )

            temp_small = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_small.close()
            resize_with_padding(
                img_path2, scale_factor, temp_small.name, image_size=image_size
            )
            img2_path = rotate_image(
                temp_small.name,
                angle,
                os.path.join(temp_dir, f"omniglot_{idx}_img2.png"),
                image_size=image_size,
            )
            os.unlink(temp_small.name)

        # Run inference
        stitched_tmp = None
        try:
            if is_llava:
                stitched_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                stitched_tmp.close()
                stitch_two_images_side_by_side(img1_path, img2_path, stitched_tmp.name)
                prompt_for_model = (
                    "You are given a single composite image with two panels placed side by side.\n\n"
                    + prompt_template
                )
                infer_payload = {
                    "image_path": stitched_tmp.name,
                    "text_prompt": prompt_for_model,
                }
            else:
                infer_payload = {
                    "image_paths": [img1_path, img2_path],
                    "text_prompt": prompt_template,
                    "max_pixels": image_size * image_size * 2,  # Two images
                }
            response = safe_infer(model, infer_payload)
        except Exception as e:
            logger.warning(f"Error during model call for Omniglot sample {idx}: {e}")
            prediction = "unknown"
            response_clean = str(e)
        else:
            parsed = parse_response(response)
            prediction = parsed["answer"]
            response_clean = parsed["response_clean"]
        finally:
            if stitched_tmp is not None and os.path.exists(stitched_tmp.name):
                os.unlink(stitched_tmp.name)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Record results
        is_correct = (prediction == "yes" and is_positive) or (
            prediction == "no" and not is_positive
        )
        results.append(
            {
                "dataset": "omniglot",
                "script_name": script_name,
                "char_id": char_id,
                "is_positive": is_positive,
                "scale_factor": scale_factor,
                "angle": angle,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
        )

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = os.path.join(exp_dir, f"{model_name}_size_rotation.csv")
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
    generalization_gap = alphabet_acc - omniglot_acc

    logger.info(
        f"Alphabet Accuracy: {alphabet_acc:.2f}% ({alphabet_df['is_correct'].sum()}/{len(alphabet_df)})"
    )
    logger.info(
        f"Omniglot Accuracy: {omniglot_acc:.2f}% ({omniglot_df['is_correct'].sum()}/{len(omniglot_df)})"
    )
    logger.info(f"Generalization Gap (Δ): {generalization_gap:.2f}%")

    # Accuracy by scale factor
    for scale in scale_factors:
        alphabet_scale_df = alphabet_df[alphabet_df["scale_factor"] == scale]
        omniglot_scale_df = omniglot_df[omniglot_df["scale_factor"] == scale]
        alphabet_scale_acc = (
            alphabet_scale_df["is_correct"].mean() * 100
            if len(alphabet_scale_df) > 0
            else 0
        )
        omniglot_scale_acc = (
            omniglot_scale_df["is_correct"].mean() * 100
            if len(omniglot_scale_df) > 0
            else 0
        )
        logger.info(
            f"Scale {scale:.1f} - Alphabet: {alphabet_scale_acc:.2f}%, Omniglot: {omniglot_scale_acc:.2f}%"
        )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    datasets = ["Alphabet\n(High Semantic)", "Omniglot\n(Low Semantic)"]
    accuracies = [alphabet_acc, omniglot_acc]
    n_samples = [len(alphabet_df), len(omniglot_df)]
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
    gap_color = (
        "red"
        if generalization_gap > 20
        else "orange" if generalization_gap > 10 else "green"
    )
    total_samples = len(alphabet_df) + len(omniglot_df)
    bars = ax2.barh(
        ["Generalization Gap (Δ)"],
        [generalization_gap],
        color=gap_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.axvline(0, color="black", linestyle="-", linewidth=1)
    ax2.axvline(
        20, color="red", linestyle="--", linewidth=1, label="Critical Threshold (20%)"
    )
    ax2.set_xlabel("Accuracy Difference (%)", fontsize=12)
    ax2.text(
        generalization_gap / 2 if generalization_gap > 0 else generalization_gap / 2,
        bars[0].get_y() + bars[0].get_height() / 2,
        f"Δ = {generalization_gap:.1f}%\n(n={total_samples})",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="white" if abs(generalization_gap) > 10 else "black",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. Accuracy by scale factor
    ax3 = axes[1, 0]
    scale_labels = [f"{s:.1f}" for s in scale_factors]
    x = np.arange(len(scale_labels))
    width = 0.35

    alphabet_scale_accs = [
        (
            alphabet_df[alphabet_df["scale_factor"] == s]["is_correct"].mean() * 100
            if len(alphabet_df[alphabet_df["scale_factor"] == s]) > 0
            else 0
        )
        for s in scale_factors
    ]
    omniglot_scale_accs = [
        (
            omniglot_df[omniglot_df["scale_factor"] == s]["is_correct"].mean() * 100
            if len(omniglot_df[omniglot_df["scale_factor"] == s]) > 0
            else 0
        )
        for s in scale_factors
    ]
    alphabet_scale_n = [
        len(alphabet_df[alphabet_df["scale_factor"] == s]) for s in scale_factors
    ]
    omniglot_scale_n = [
        len(omniglot_df[omniglot_df["scale_factor"] == s]) for s in scale_factors
    ]

    bars1 = ax3.bar(
        x - width / 2,
        alphabet_scale_accs,
        width,
        label="Alphabet",
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax3.bar(
        x + width / 2,
        omniglot_scale_accs,
        width,
        label="Omniglot",
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_ylabel("Accuracy (%)", fontsize=12)
    ax3.set_xlabel("Scale Factor", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scale_labels)
    ax3.set_ylim(0, 100)
    ax3.axhline(
        50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    for i, (bar, n) in enumerate(zip(bars1, alphabet_scale_n)):
        height = bar.get_height()
        if height > 0 or n > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%\n(n={n})",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    for i, (bar, n) in enumerate(zip(bars2, omniglot_scale_n)):
        height = bar.get_height()
        if height > 0 or n > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%\n(n={n})",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    # 4. Accuracy by rotation angle (binned)
    ax4 = axes[1, 1]
    angle_bins = [0, 90, 180, 270, 360]
    angle_labels = ["0-90°", "90-180°", "180-270°", "270-360°"]

    alphabet_df["angle_bin"] = pd.cut(
        alphabet_df["angle"], bins=angle_bins, labels=angle_labels
    )
    omniglot_df["angle_bin"] = pd.cut(
        omniglot_df["angle"], bins=angle_bins, labels=angle_labels
    )

    alphabet_angle_acc = alphabet_df.groupby("angle_bin")["is_correct"].mean() * 100
    omniglot_angle_acc = omniglot_df.groupby("angle_bin")["is_correct"].mean() * 100
    alphabet_angle_n = alphabet_df.groupby("angle_bin").size()
    omniglot_angle_n = omniglot_df.groupby("angle_bin").size()

    x = np.arange(len(angle_labels))
    bars1 = ax4.bar(
        x - width / 2,
        [alphabet_angle_acc.get(label, 0) for label in angle_labels],
        width,
        label="Alphabet",
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax4.bar(
        x + width / 2,
        [omniglot_angle_acc.get(label, 0) for label in angle_labels],
        width,
        label="Omniglot",
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_ylabel("Accuracy (%)", fontsize=12)
    ax4.set_xlabel("Rotation Angle Range", fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(angle_labels, rotation=45, ha="right")
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    for i, (bar, label) in enumerate(zip(bars1, angle_labels)):
        height = bar.get_height()
        n = alphabet_angle_n.get(label, 0)
        if height > 0 or n > 0:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%\n(n={n})",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
    for i, (bar, label) in enumerate(zip(bars2, angle_labels)):
        height = bar.get_height()
        n = omniglot_angle_n.get(label, 0)
        if height > 0 or n > 0:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.1f}%\n(n={n})",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f"{model_name}_size_rotation_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {plot_path}")
    plt.close()

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {exp_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Illusion Experiment")
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
        help="(Legacy) repo base dir; image loading uses --rotation_data_dir",
    )
    parser.add_argument(
        "--rotation_data_dir",
        type=str,
        default=DEFAULT_ROTATION_DATA_DIR,
        help=(
            "Python dir with images_original/ and images_rotated/ "
            f"(default: {DEFAULT_ROTATION_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl",
        help="Model id (e.g. allenai/Molmo2-8B, Qwen/Qwen2.5-VL-32B-Instruct)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="rotation",
        choices=["rotation", "size_rotation"],
        help="Experiment type: 'rotation' (default) or 'size_rotation'",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="(Unused for rotation) kept for backward compatibility",
    )
    parser.add_argument(
        "--positive_ratio",
        type=float,
        default=0.5,
        help="(Unused for rotation) both pos/neg run per char×angle",
    )
    parser.add_argument(
        "--omniglot_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Optional Omniglot script folders under images_original (default: all). "
        "English is the alphabet dataset (times_new_roman), not an Omniglot folder.",
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
        "--resume_csv",
        type=str,
        default=None,
        help="Optional CSV to resume from (skip existing entries)",
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="Scale factors for size_rotation experiment (default: 0.5 0.7 0.9)",
    )

    args = parser.parse_args()

    model_config = {"model_name": args.model}

    if args.experiment == "rotation":
        run_spatial_illusion_experiment(
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
            omniglot_scripts=args.omniglot_scripts if args.omniglot_scripts else None,
            resume_csv=args.resume_csv,
            rotation_data_dir=args.rotation_data_dir,
        )
    elif args.experiment == "size_rotation":
        run_size_rotation_experiment(
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
        )
