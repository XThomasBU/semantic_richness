import torch
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .base_encoder import BaseVisionTower


class QwenVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, config, delay_load=False):
        super(QwenVisionTower, self).__init__(vision_tower_name, config, delay_load)

        self._config = config
        self.select_feature = self._config.get("mm_vision_select_feature", "patch")
        self.select_layer = self._config.get("mm_vision_select_layer", -2)

        if not self.delay_load:
            self.load_model()
        else:
            from transformers import Qwen2_5_VLConfig

            self.cfg_only = Qwen2_5_VLConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return

        print(f"Loading full Qwen VL model: {self.vision_tower_name}")

        self.vision_tower = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vision_tower_name, torch_dtype=self._config.get("torch_dtype", "auto")
        )
        print("Full model loaded.")

        print("Loading processor using AutoProcessor...")
        self.image_processor = AutoProcessor.from_pretrained(
            self.vision_tower_name, use_fast=False
        ).image_processor
        print("Processor loaded.")

        self.vision_tower.requires_grad_(
            self._config.get("unfreeze_mm_vision_tower", False)
        )
        self.is_loaded = True

    def _feature_select(self, image_features):
        if self.select_feature == "patch":
            features = image_features[:, 1:]
        elif self.select_feature == "patch_mean":
            features = image_features[:, 1:].mean(dim=1)
        elif self.select_feature == "cls_patch":
            features = image_features
        elif self.select_feature == "cls":
            features = image_features[:, 0]
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return features

    @staticmethod
    def _unwrap_visual_output(vision_out):
        """Handle raw tensor (older transformers) or BaseModelOutputWithPooling (>=5.x)."""
        if isinstance(vision_out, torch.Tensor):
            return vision_out
        pooler = getattr(vision_out, "pooler_output", None)
        if pooler is not None:
            if isinstance(pooler, (list, tuple)):
                if len(pooler) == 1:
                    return pooler[0]
                return torch.stack([t for t in pooler], dim=0)
            return pooler
        last_hidden = getattr(vision_out, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        raise TypeError(
            f"Unexpected vision tower output type: {type(vision_out).__name__}"
        )

    def _forward(self, images, **kwargs):
        """
        Runs the forward pass.
        'images' is the pixel_values tensor.
        'kwargs' contains 'grid_thw'.
        """
        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None:
            raise ValueError("QwenVisionTower's _forward method requires 'grid_thw'.")

        self.load_model()

        with torch.set_grad_enabled(
            self._config.get("unfreeze_mm_vision_tower", False)
        ):

            vision_out = self.vision_tower.model.visual(
                images.to(device=self.device, dtype=self.dtype),
                grid_thw=grid_thw.to(device=self.device),
            )
            image_features = self._unwrap_visual_output(vision_out)

            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(0)

            final_features = self._feature_select(image_features)

            return final_features.to(images.dtype)

    @property
    def dtype(self):
        try:
            return self.vision_tower.dtype
        except AttributeError:
            try:
                return getattr(self.cfg_only, "torch_dtype", torch.float32)
            except AttributeError:
                return torch.float32

    @property
    def device(self):
        try:
            return self.vision_tower.device
        except AttributeError:
            return torch.device("cpu")

    @property
    def num_patches(self):
        try:
            config = self.vision_tower.config.vision_config
            return (config.image_size // config.patch_size) ** 2
        except AttributeError:
            try:
                config = self.cfg_only.vision_config
                return (config.image_size // config.patch_size) ** 2
            except AttributeError:
                return (224 // 14) ** 2
