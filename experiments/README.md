# Experiments

All experiments for evaluating VLM visual invariance on the Omniglot dataset.

## Scale Illusion (`scale/`)

Tests whether VLMs can recognize characters across scale changes. Presents pairs of images at different sizes and asks the model if they are the same character.

```bash
# Local models (Qwen, LLaVA, etc.)
python -m experiments.scale.scale_illusion --model Qwen/Qwen2.5-VL-7B-Instruct --output_dir results/scale

# API models (Gemini, GPT)
python -m experiments.scale.scale_illusion --model gemini-2.5-pro --output_dir results/scale

# PACS dataset variant
python -m experiments.scale.scale_illusion_pacs --model Qwen/Qwen2.5-VL-72B-Instruct --pacs_dir /path/to/pacs

# Bootstrap confidence intervals
python -m experiments.scale.scale_illusion_bootstrap_ci --results_root results/
```

Analysis: `analyze_scale_illusion.py`, `compute_scale_illusion_metrics.py`

## Identity Illusion (`identity/`)

Tests whether VLMs can determine if two images depict the same character (without rotation). Evaluates identity recognition across familiar vs unfamiliar scripts.

```bash
# Spatial illusion (identity across scripts)
python -m experiments.identity.spatial_illusion --model Qwen/Qwen2.5-VL-7B-Instruct --output_dir results/identity

# PACS variant
python -m experiments.identity.spatial_illusion_pacs --model Qwen/Qwen2.5-VL-7B-Instruct --data_dir /path/to/pacs
```

Analysis: `analyze_identity_illusion_failures.py`, `analyze_spatial_illusion_by_script.py`, `analyze_spatial_illusion_categories.py`

## Rotation (`rotation/`)

Tests whether VLMs can determine if one image is a rotation of another. Uses Omniglot characters rotated at various angles.

```bash
# Rotation recognition
python -m experiments.rotation.rotation_recog --model qwen --script Armenian --output_dir results/rotation

# Generate rotated images
python -m experiments.rotation.rotator

# Bootstrap confidence intervals
python -m experiments.rotation.rotation_bootstrap_ci --omniglot_root DATA/omniglot-master
```

Analysis: `rotation_response_analysis.py`, `rotation_interp.py` (feature interpolation)

## Linear Probe (`linear_probe/`)

Trains linear classifiers on frozen vision encoder features to probe rotation angle representations.

```bash
# Precompute features
python -m experiments.linear_probe.precompute_features --encoder clip --dataset omniglot

# Run probe
python -m experiments.linear_probe.run_probe --features_path results/linear_probe/features/*.pt

# Character holdout variant
python -m experiments.linear_probe.run_probe_char_holdout --features_path results/linear_probe/features/*.pt
```

## Cosine Similarity (`cosine_similarity/`)

Measures how vision encoder embeddings change under rotation by computing cosine similarity between original and rotated character images.

```bash
# Demo (from repo root)
python demo/run_cosine_similarity.py --encoder clip dino siglip
```

Supported encoders: `clip`, `dino`, `siglip`, `qwen` (GPU required)

## Embedding Analysis (`embedding_analysis/`)

Analyzes effective rank, dimensionality, and t-SNE/NMI of vision encoder embeddings across scripts.

```bash
# Effective rank analysis
python -m experiments.embedding_analysis.effective_rank_analysis --model Qwen/Qwen2.5-VL-7B-Instruct

# Plot effective rank by layer
python -m experiments.embedding_analysis.plot_erank_by_layer --input_dir results/erank/
```

## Analysis & Plotting (`analysis/`)

Cross-experiment plotting and aggregation scripts:
- `final_plots.py` — paper figures
- `model_category_perf.py` — model performance comparison
- `aggregate_pacs_domain_scale_performance.py` — PACS domain aggregation
- `dataset_snapshot_grids.py` — dataset visualization grids
