import torch
import torch.nn as nn
from torchvision import transforms as T

import random


class RandomIntensityScale:
    def __init__(self, factor_range=(0.9, 1.1)):
        self.factor_range = factor_range

    def __call__(self, tensor):
        """
        :param tensor: Tensor of shape (C, H, W)
        """
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        return tensor * factor


class GaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        :param tensor: Tensor of shape (C, H, W)
        """
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


def get_augmentation_pipeline(
        p_affine=0.7,
        p_scale=0.7,
        p_noise=0.5,
        p_erase=0.5,
        translate_range=(0.05, 0.05),
        scale_factor=(0.9, 1.1),
        noise_std=0.01,
        erase_scale_range=(0.02, 0.1)
):
    transforms_list = [
        T.RandomApply(
            nn.ModuleList([
                T.RandomAffine(degrees=0, translate=translate_range)
            ]),
            p=p_affine
        ),
        T.RandomApply(
            [
                RandomIntensityScale(factor_range=scale_factor)
            ],
            p=p_scale
        ),
        T.RandomApply(
            [
                GaussianNoise(mean=0., std=noise_std)
            ],
            p=p_noise
        ),
        T.RandomErasing(
            p=p_erase,
            scale=erase_scale_range,
            ratio=(0.5, 2.0),
            value=0
        )
    ]

    return T.Compose(transforms_list)