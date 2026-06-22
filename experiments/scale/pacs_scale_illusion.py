"""
Experiment: PACS Scale Illusion (Scale Invariance vs. Domain/Category Priors)

Goal: Test whether the model matches images based on scale-invariant geometry
or relies on domain/category priors in PACS (art_painting, cartoon, photo, sketch).
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
from typing import List, Tuple, Dict

import pandas as pd
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
SCALE_FACTORS = [0.1, 0.3, 0.5, 0.9]

PROMPT_DIRECT = (
    "You are given two images. Decide whether they depict the same underlying object, "
    "allowing for significant differences in size, scale, or resolution. "
    "Answer with exactly YES or NO."
)


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
                wait_time = (2 ** attempt) + random.random() * 2
                print(f"[WARN] Server overloaded (attempt {attempt}/{max_retries}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 1 + random.random()
                print(f"[WARN] Error during inference (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Model unavailable after {max_retries} retries. Aborting inference.")


def parse_response(response_text: str) -> Dict[str, str]:
    if not response_text:
        return {"answer": "unknown", "response_clean": ""}
    if isinstance(response_text, list):
        response_text = response_text[0] if response_text else ""
    response_clean = str(response_text)
    text_for_matching = str(response_text).lower().strip()
    if re.search(r"\byes\b|\btrue\b|\bsame\b", text_for_matching):
        return {"answer": "yes", "response_clean": response_clean}
    if re.search(r"\bno\b|\bfalse\b|\bdifferent\b", text_for_matching):
        return {"answer": "no", "response_clean": response_clean}
    return {"answer": "unknown", "response_clean": response_clean}


def resize_with_padding(image_path: str, scale_factor: float, output_path: str = None, image_size: int = 336) -> str:
    img = Image.open(image_path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    new_char_size = int(image_size * scale_factor)
    img_small = img.resize((new_char_size, new_char_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    paste_x = (image_size - new_char_size) // 2
    paste_y = (image_size - new_char_size) // 2
    canvas.paste(img_small, (paste_x, paste_y))
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"resized_{random.randint(0, 999999)}.png")
    canvas.save(output_path, "PNG")
    return output_path


def load_pacs_images(pacs_dir: str, max_per_category: int = 10) -> List[Tuple[str, str, str]]:
    """
    Load PACS images from domain_category directories.
    Returns list of (domain, category, image_path).
    """
    base_path = Path(pacs_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"PACS directory not found: {base_path}")

    images = []
    for domain_category_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):
        name = domain_category_dir.name
        if "_" not in name:
            continue
        domain, category = name.rsplit("_", 1)
        category_dir = domain_category_dir / category
        if not category_dir.exists():
            continue
        files = sorted(
            list(category_dir.rglob("*.png"))
            + list(category_dir.rglob("*.jpg"))
            + list(category_dir.rglob("*.jpeg"))
        )
        for img_path in files[:max_per_category]:
            images.append((domain, category, str(img_path)))
    return images


def create_pacs_sanity_check(
    pacs_dir: str,
    output_path: str,
    num_examples: int = 6,
    image_size: int = 336,
    scale_factors: List[float] = None,
    max_per_category: int = 10,
):
    if scale_factors is None:
        scale_factors = SCALE_FACTORS
    pacs_images = load_pacs_images(pacs_dir, max_per_category=max_per_category)
    if len(pacs_images) < 2:
        raise ValueError("Not enough PACS images for sanity check")

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 2.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_examples):
        scale_factor = random.choice(scale_factors)
        domain, category, img_path = random.choice(pacs_images)
        is_positive = i % 2 == 0

        img1 = Image.open(img_path).convert("RGB")
        if img1.size != (image_size, image_size):
            img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)

        if is_positive:
            temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_scaled.close()
            resize_with_padding(img_path, scale_factor, temp_scaled.name, image_size=image_size)
            img2 = Image.open(temp_scaled.name).convert("RGB")
            os.unlink(temp_scaled.name)
            label = f"PACS: Positive\n{domain}/{category} (scale={scale_factor:.1f})"
        else:
            same_domain = [(d, c, p) for d, c, p in pacs_images if d == domain and c != category]
            if same_domain:
                domain2, category2, img_path2 = random.choice(same_domain)
            else:
                domain2, category2, img_path2 = random.choice([x for x in pacs_images if x[2] != img_path])
            temp_scaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_scaled.close()
            resize_with_padding(img_path2, scale_factor, temp_scaled.name, image_size=image_size)
            img2 = Image.open(temp_scaled.name).convert("RGB")
            os.unlink(temp_scaled.name)
            label = f"PACS: Negative\n{domain}/{category} vs {domain2}/{category2} (scale={scale_factor:.1f})"

        img1_display = img1.resize((150, 150), Image.Resampling.LANCZOS)
        img2_display = img2.resize((150, 150), Image.Resampling.LANCZOS)
        axes[i, 0].imshow(img1_display)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img2_display)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 0].text(0.5, -0.1, "Original\n(Full Size)", transform=axes[i, 0].transAxes,
                           ha="center", fontsize=10, fontweight="bold")
            axes[i, 1].text(0.5, -0.1, "Scaled\n(With Padding)", transform=axes[i, 1].transAxes,
                           ha="center", fontsize=10, fontweight="bold")
        axes[i, 1].text(1.1, 0.5, label, transform=axes[i, 1].transAxes,
                       fontsize=9, verticalalignment="center",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PACS sanity check visualization saved to: {output_path}")


def run_pacs_scale_experiment(
    pacs_dir: str,
    model_config: Dict[str, str] = {"model_name": "Qwen/Qwen2.5-VL-32B-Instruct"},
    positive_ratio: float = 0.5,
    output_dir: str = "./results",
    log_path: str = None,
    image_size: int = 336,
    rate_limit_delay: float = 0.0,
    scale_factors: List[float] = None,
    max_per_category: int = 10,
):
    if scale_factors is None:
        scale_factors = SCALE_FACTORS
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("PACS SCALE ILLUSION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {model_config.get('model_name', 'unknown')}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Scale factors: {scale_factors}")
    logger.info(f"Max per category: {max_per_category}")
    logger.info(f"Output directory: {output_dir}")

    exp_dir = os.path.join(output_dir, "pacs_scale_illusion")
    os.makedirs(exp_dir, exist_ok=True)
    sanity_dir = os.path.join(exp_dir, "sanity_check")
    os.makedirs(sanity_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="pacs_scale_illusion_")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        logger.info("Initializing model...")
        model_name = model_config.get("model_name", "unknown")
        model = InferenceModel(model_name)
        logger.info(f"Model initialized: {model_name}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise e

    logger.info("Creating PACS sanity check visualization...")
    sanity_path = os.path.join(sanity_dir, "pacs_scale_illusion_sanity_check.png")
    create_pacs_sanity_check(
        pacs_dir=pacs_dir,
        output_path=sanity_path,
        num_examples=6,
        image_size=image_size,
        scale_factors=scale_factors,
        max_per_category=max_per_category,
    )

    logger.info("Loading PACS images...")
    pacs_images = load_pacs_images(pacs_dir, max_per_category=max_per_category)
    logger.info(f"Loaded {len(pacs_images)} PACS images")
    if len(pacs_images) < 2:
        raise ValueError("Not enough PACS images")

    results = []
    samples = [(domain, category, img_path, scale)
               for domain, category, img_path in pacs_images
               for scale in scale_factors]
    logger.info(f"Using all {len(samples)} PACS samples (images × scales)")

    for idx, (domain, category, img_path, scale_factor) in enumerate(tqdm(samples, desc="PACS")):
        is_positive = random.random() < positive_ratio
        img1 = Image.open(img_path).convert("RGB")
        if img1.size != (image_size, image_size):
            img1 = img1.resize((image_size, image_size), Image.Resampling.LANCZOS)
        temp_img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_img1.close()
        img1.save(temp_img1.name, "PNG")
        img1_path = temp_img1.name

        if is_positive:
            temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img2.close()
            resize_with_padding(img_path, scale_factor, temp_img2.name, image_size=image_size)
            img2_path = temp_img2.name
        else:
            same_domain = [(d, c, p) for d, c, p in pacs_images if d == domain and c != category]
            if same_domain:
                domain2, category2, img_path2 = random.choice(same_domain)
            else:
                domain2, category2, img_path2 = random.choice([x for x in pacs_images if x[2] != img_path])
            temp_img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img2.close()
            resize_with_padding(img_path2, scale_factor, temp_img2.name, image_size=image_size)
            img2_path = temp_img2.name

        try:
            response = safe_infer(
                model,
                {
                    "image_paths": [img1_path, img2_path],
                    "text_prompt": PROMPT_DIRECT,
                    "max_pixels": image_size * image_size * 2,
                },
            )
            parsed = parse_response(response)
            prediction = parsed["answer"]
            response_clean = parsed["response_clean"]
        except Exception as e:
            logger.warning(f"Error in inference for PACS sample {idx}: {e}")
            prediction = "unknown"
            response_clean = str(e)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        os.unlink(temp_img1.name)
        os.unlink(temp_img2.name)

        is_correct = (prediction == "yes" and is_positive) or (prediction == "no" and not is_positive)
        results.append(
            {
                "dataset": "pacs",
                "domain": domain,
                "category": category,
                "is_positive": is_positive,
                "scale_factor": scale_factor,
                "prediction": prediction,
                "is_correct": is_correct,
                "response": response_clean,
            }
        )

    shutil.rmtree(temp_dir)

    df = pd.DataFrame(results)
    csv_path = os.path.join(exp_dir, "pacs_scale_illusion.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Analysis + plots
    pacs_acc = df["is_correct"].mean() * 100
    logger.info(f"PACS Accuracy: {pacs_acc:.2f}% ({df['is_correct'].sum()}/{len(df)})")

    # Accuracy by domain
    domain_df = df.groupby("domain")["is_correct"].mean().reset_index(name="accuracy")
    domain_df["accuracy"] *= 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=domain_df, x="domain", y="accuracy", hue="domain", palette="viridis", ax=ax, legend=False)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_accuracy_by_domain.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy by category
    cat_df = df.groupby("category")["is_correct"].mean().reset_index(name="accuracy")
    cat_df["accuracy"] *= 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cat_df, x="category", y="accuracy", hue="category", palette="mako", ax=ax, legend=False)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_accuracy_by_category.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy by scale factor
    scale_df = df.groupby("scale_factor")["is_correct"].mean().reset_index(name="accuracy")
    scale_df["accuracy"] *= 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=scale_df, x="scale_factor", y="accuracy", marker="o", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Scale Factor")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_accuracy_by_scale.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy heatmap: domain x category
    heatmap_df = (
        df.groupby(["domain", "category"])["is_correct"]
        .mean()
        .reset_index()
        .pivot(index="domain", columns="category", values="is_correct")
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_df, ax=ax, cmap="viridis", vmin=0, vmax=100, cbar_kws={"label": "Accuracy (%)"})
    ax.set_xlabel("Category")
    ax.set_ylabel("Domain")
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_accuracy_domain_category_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy by domain, broken down by category
    domain_cat_df = df.groupby(["domain", "category"])["is_correct"].mean().reset_index(name="accuracy")
    domain_cat_df["accuracy"] *= 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=domain_cat_df, x="category", y="accuracy", hue="domain", palette="Set2", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_accuracy_by_domain_and_category.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Recall (TPR) and Specificity (TNR) by domain
    metrics_rows = []
    for domain, group in df.groupby("domain"):
        pos_df = group[group["is_positive"] == True]
        neg_df = group[group["is_positive"] == False]
        tpr = (pos_df["is_correct"].mean() * 100) if len(pos_df) > 0 else 0
        tnr = (neg_df["is_correct"].mean() * 100) if len(neg_df) > 0 else 0
        metrics_rows.append({"domain": domain, "metric": "Recall (TPR)", "value": tpr})
        metrics_rows.append({"domain": domain, "metric": "Specificity (TNR)", "value": tnr})
    metrics_df = pd.DataFrame(metrics_rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=metrics_df, x="domain", y="value", hue="metric", palette="Set2", ax=ax)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_recall_specificity_by_domain.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Recall and Specificity by category
    cat_metrics = []
    for category, group in df.groupby("category"):
        pos_df = group[group["is_positive"] == True]
        neg_df = group[group["is_positive"] == False]
        tpr = (pos_df["is_correct"].mean() * 100) if len(pos_df) > 0 else 0
        tnr = (neg_df["is_correct"].mean() * 100) if len(neg_df) > 0 else 0
        cat_metrics.append({"category": category, "metric": "Recall (TPR)", "value": tpr})
        cat_metrics.append({"category": category, "metric": "Specificity (TNR)", "value": tnr})
    cat_metrics_df = pd.DataFrame(cat_metrics)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cat_metrics_df, x="category", y="value", hue="metric", palette="Set2", ax=ax)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_recall_specificity_by_category.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Recall and Specificity by domain and category (facet)
    dom_cat_metrics = []
    for (domain, category), group in df.groupby(["domain", "category"]):
        pos_df = group[group["is_positive"] == True]
        neg_df = group[group["is_positive"] == False]
        tpr = (pos_df["is_correct"].mean() * 100) if len(pos_df) > 0 else 0
        tnr = (neg_df["is_correct"].mean() * 100) if len(neg_df) > 0 else 0
        dom_cat_metrics.append({"domain": domain, "category": category, "metric": "Recall (TPR)", "value": tpr})
        dom_cat_metrics.append({"domain": domain, "category": category, "metric": "Specificity (TNR)", "value": tnr})
    dom_cat_df = pd.DataFrame(dom_cat_metrics)
    g = sns.catplot(
        data=dom_cat_df,
        x="category",
        y="value",
        hue="metric",
        col="domain",
        kind="bar",
        col_wrap=2,
        palette="Set2",
        height=3.5,
        aspect=1.1,
        sharey=True,
    )
    g.set_axis_labels("Category", "Rate (%)")
    for ax in g.axes.flatten():
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(0, 100)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_recall_specificity_by_domain_category.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Recall and Specificity by scale factor
    scale_metrics = []
    for scale_factor, group in df.groupby("scale_factor"):
        pos_df = group[group["is_positive"] == True]
        neg_df = group[group["is_positive"] == False]
        tpr = (pos_df["is_correct"].mean() * 100) if len(pos_df) > 0 else 0
        tnr = (neg_df["is_correct"].mean() * 100) if len(neg_df) > 0 else 0
        scale_metrics.append({"scale_factor": scale_factor, "metric": "Recall (TPR)", "value": tpr})
        scale_metrics.append({"scale_factor": scale_factor, "metric": "Specificity (TNR)", "value": tnr})
    scale_metrics_df = pd.DataFrame(scale_metrics)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=scale_metrics_df, x="scale_factor", y="value", hue="metric", marker="o", ax=ax)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Scale Factor")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "pacs_recall_specificity_by_scale.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"PACS plots saved to: {exp_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACS Scale Illusion Experiment")
    parser.add_argument("--pacs_dir", type=str, required=True, help="Base directory containing pacs_images")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                       choices=[
                           "Qwen/Qwen2.5-VL-7B-Instruct",
                           "Qwen/Qwen2.5-VL-32B-Instruct",
                           "Qwen/Qwen2.5-VL-72B-Instruct",
                           "qwen2.5-vl",
                           "gemini-2.5-pro",
                       ],
                       help="Model to use")
    parser.add_argument("--positive_ratio", type=float, default=0.5, help="Ratio of positive pairs (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument("--image_size", type=int, default=336, help="Image size (default: 336)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0, help="Delay between API calls (seconds)")
    parser.add_argument("--scale_factors", type=float, nargs="+", default=SCALE_FACTORS, help="Scale factors")
    parser.add_argument("--max_per_category", type=int, default=10, help="Max images per category")

    args = parser.parse_args()
    model_config = {"model_name": args.model}
    run_pacs_scale_experiment(
        pacs_dir=args.pacs_dir,
        model_config=model_config,
        positive_ratio=args.positive_ratio,
        output_dir=args.output_dir,
        log_path=args.log_path,
        image_size=args.image_size,
        rate_limit_delay=args.rate_limit_delay,
        scale_factors=args.scale_factors,
        max_per_category=args.max_per_category,
    )
