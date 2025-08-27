import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from dataset.transform import *
from torchvision import transforms as T

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
Image.MAX_IMAGE_PIXELS = None 

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.transform = transform

        # load ids
        if mode in ['train_l', 'train_u']:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids = (self.ids * math.ceil(nsample / len(self.ids)))[:nsample]
        else:
            val_path = os.path.join('splits', name, 'val.txt')
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def _augment_strong(self, img):
        """Strong augmentation for unlabeled images."""
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = blur(img, p=0.5)
        return img

    def __getitem__(self, idx):
        id_line = self.ids[idx]
        img_path = os.path.join(self.root, id_line.split(' ')[0])
        img = Image.open(img_path).convert('RGB')

        # Mask
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask_path = os.path.join(self.root, id_line.split(' ')[1])
            mask = Image.open(mask_path)

        # Validation
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask

        # Train augmentations
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        img, mask = bflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        # Unlabeled strong augmentations
        img_w, img_s1, img_s2 = img.copy(), img.copy(), img.copy()
        img_s1 = self._augment_strong(img_s1)
        img_s2 = self._augment_strong(img_s2)
        cutmix_box = obtain_cutmix_box(img_s1.size[0], p=0.5)

        # Ignore mask
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = self.ignore_value

        return normalize(img_w), img_s1, normalize(img_s2), ignore_mask, cutmix_box
