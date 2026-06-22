import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import create_model_from_pretrained

from .base_encoder import BaseVisionTower, ProcessorWrapper


def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14": "hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP": "hf-hub:timm/ViT-SO400M-14-SigLIP",
        "google/siglip-so400m-patch14-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
    }

    res = 384 if "384" in model_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SiglipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(SiglipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_res_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        clip_model, processor = create_model_from_pretrained(self.vision_tower_name)

        self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        self.image_processor = ProcessorWrapper(
            processor, height=self._image_size, width=self._image_size
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _feature_select(self, image_features):
        if self.select_feature == "multihead_attention_pool":
            return self.vision_tower.attn_pool(image_features)
        if self.select_feature == "patch_mean":
            return image_features.mean(dim=1)
        if self.select_feature == "patch":
            return image_features
        if self.select_feature == "cls":
            return image_features[:, 0]
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            image_features = image_features.permute(0, 2, 3, 1).contiguous()
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, **kwargs):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(
                images.to(device=self.device, dtype=self.dtype)
            )
            interp_features = self.interpolate(image_features)
            return self._feature_select(interp_features)
