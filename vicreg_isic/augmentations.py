# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 0.5 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class CropMainAxis:
    def __init__(self,margin=1.1) -> None:
        self.margin = margin

    def __call__(self, img):
        print(len(img))
        c,w, h = img.size()
        return transforms.CenterCrop(self.margin*min(w, h))(img)


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
            
isic_mean = torch.tensor([0.6116, 0.4709, 0.4692])
isic_std = torch.tensor([0.2560, 0.2217, 0.2273])

class TrainTransform(object):
    def __init__(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                CropMainAxis(),
                transforms.Resize((224, 224)),
                transforms.ToPILImage(),
                # transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(360),
                GaussianBlur(p=1.),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.2, hue=0.)
                    ], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(isic_mean, isic_std),
                ])  

        self.transform_prime = transforms.Compose(
            [
                transforms.ToTensor(),
                CropMainAxis(),
                transforms.Resize((224, 224)),
                transforms.ToPILImage(),
                # transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(360),
                GaussianBlur(p=0.1),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.)
                    ], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(isic_mean, isic_std)   
            ])  

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
