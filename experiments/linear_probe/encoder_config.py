"""Shared encoder configs for feature extraction and linear probing."""

ENCODER_ARGS = {
    "clip": {
        "mm_vision_select_layer": -1,
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "patch_mean",
    },
    "dino": {
        "mm_vision_select_layer": -1,
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "patch_mean",
    },
    "diffusion": {
        "mm_vision_select_layer": 1,
        "unfreeze_mm_vision_tower": False,
        "model_name": "stabilityai/stable-diffusion-2-1",
        "diffusion_step_type": "onestep",
        "time_step": 25,
    },
    "dit": {
        "mm_vision_select_layer": 13,
        "model_name": "facebook/DiT-XL-2-512",
        "time_step": 25,
        "unfreeze_mm_vision_tower": False,
    },
    "qwen": {
        "mm_vision_select_layer": -1,
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "cls",
    },
    "siglip": {
        "mm_vision_select_layer": -1,
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "multihead_attention_pool",
    },
}

DEFAULT_MODELS = {
    "clip": "openai/clip-vit-large-patch14-336",
    "dino": "facebook/dinov2-large",
    "diffusion": "stabilityai/stable-diffusion-2-1",
    "dit": "facebook/DiT-XL-2-512",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "siglip": "google/siglip-so400m-patch14-384",
}


def resolve_model_name(encoder: str, model_name: str | None) -> str:
    return model_name or DEFAULT_MODELS[encoder]


def resolve_encoder_args(encoder: str, model_name: str | None) -> dict:
    args = dict(ENCODER_ARGS[encoder])
    if encoder in ("diffusion", "dit", "qwen") and model_name:
        args["model_name"] = model_name
    return args
