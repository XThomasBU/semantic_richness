# Visual Invariance

Evaluating Vision-Language Models on geometric visual invariance using the Omniglot dataset. Tests whether VLMs truly understand scale, identity, and rotation, or rely on semantic priors.

## Structure

```
experiments/
  scale/              Scale illusion experiments
  identity/           Identity/spatial illusion experiments
  rotation/           Rotation recognition experiments
  linear_probe/       Linear probe on frozen encoder features
  cosine_similarity/  Cosine similarity under rotation
  embedding_analysis/ Effective rank and embedding analysis
  analysis/           Cross-experiment plotting
models/               VLM inference (Qwen, Gemini, GPT, LLaVA, InternVL, Molmo)
vision_models/        Vision encoders (CLIP, DINO, SigLIP, Qwen, Diffusion)
demo/                 Runnable demo with sample data
DATA/                 Datasets (gitignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate blind
```

Create a `.env` file with API keys (for API-based models):
```
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Quick Start

Run the cosine similarity demo on 5 sample Omniglot characters:

```bash
python demo/run_cosine_similarity.py --encoder clip dino siglip
```

See [experiments/README.md](experiments/README.md) for full experiment documentation.

## Supported Models

**Local (GPU):** Qwen2.5-VL (7B/32B/72B), Qwen3-VL, LLaVA-1.5, InternVL2.5/3.5, Molmo2

**API:** Gemini 2.5 Pro, GPT-5.2

**Vision Encoders:** CLIP, DINOv2, SigLIP, Qwen-VL, Stable Diffusion
