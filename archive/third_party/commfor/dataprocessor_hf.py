import torchvision.transforms as transforms
import torch
import argparse
import PIL.Image as Image
from typing import Union, List, Optional

from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import PushToHubMixin

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CommForImageProcessor(BaseImageProcessor, PushToHubMixin):
    """
    Aspect-ratio-preserving Image Processor for Community Forensics ViT model.
    Resizes images along the shortest edge (440px for size=384, 256px for size=224)
    and extracts center-crop or multi-crop (center + 4 corners) without aspect ratio squashing.
    """
    image_processor_type = "commfor_image_processor"
    model_input_names = ["pixel_values"]

    def __init__(self, size: int = 384, use_multi_crop: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.use_multi_crop = use_multi_crop
        assert self.size in [224, 384], f"Unsupported size: {self.size}. Supported sizes are 224 and 384."

        # Shortest-edge resize and crop targets
        self.resize_short = 440 if self.size == 384 else 256
        self.crop_size = self.size

        # Transforms
        self.resize_transform = transforms.Resize(self.resize_short, interpolation=transforms.InterpolationMode.BICUBIC)
        self.center_crop = transforms.CenterCrop(self.crop_size)
        self.five_crop = transforms.FiveCrop(self.crop_size)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _process_single(self, image: Image.Image, multi_crop: bool = False) -> torch.Tensor:
        """Process a single PIL image preserving aspect ratio."""
        image = image.convert("RGB")
        # Resize shortest edge to resize_short (preserves aspect ratio)
        resized = self.resize_transform(image)

        if multi_crop:
            # 5 crops: top-left, top-right, bottom-left, bottom-right, center
            crops = self.five_crop(resized)
            return torch.stack([self.to_tensor_norm(c) for c in crops])  # (5, C, H, W)
        else:
            cropped = self.center_crop(resized)
            return self.to_tensor_norm(cropped)  # (C, H, W)

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        mode: str = "test",
        use_multi_crop: Optional[bool] = None,
        **kwargs
    ) -> dict:
        """
        Preprocess input PIL image(s) to normalized PyTorch tensor(s).
        """
        assert mode in ["test", "train", "val"], f"Unsupported mode: {mode}."
        if isinstance(images, Image.Image):
            images = [images]

        multi_crop = self.use_multi_crop if use_multi_crop is None else use_multi_crop

        tensors = [self._process_single(im, multi_crop=multi_crop) for im in images]
        if len(tensors) == 1:
            return {"pixel_values": tensors[0]}
        else:
            return {"pixel_values": torch.stack(tensors)}
