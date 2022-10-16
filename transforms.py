from dataclasses import dataclass

import torch
from torchvision import transforms as T
from timm.data import create_transform

from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.image import ImageClassificationInputTransform


@dataclass
class TimmIputTransform(ImageClassificationInputTransform):
    def __post_init__(self):
        self.train_transform = create_transform(
            self.image_size,
            is_training=True,
            vflip=0.5,
            # auto_augment="rand-m9-mstd0.5",
        )
        self.val_transform = create_transform(self.image_size, is_training=False)
        super().__post_init__()

    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    self.val_transform,
                ),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(DataKeys.INPUT, self.train_transform),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )