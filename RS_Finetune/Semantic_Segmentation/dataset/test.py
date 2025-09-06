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


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.transform = transform


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            if name == 'loveda':
                split_file = f'splits/{name}/test.txt'
            else:
                split_file = f'splits/{name}/val.txt'

            with open(split_file, 'r') as f:
                self.ids = f.read().splitlines()


    def _trainid_to_class(self, label):

        return label

    def tes_class_to_trainid(self, label):
   
        return label

    def _class_to_trainid(self, label):

        return label

    def process_mask(self, mask):
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def __len__(self):
        # return len(self.image_list)
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.open(os.path.join(self.root, id.split(' ')[1]))

        if self.name == 'loveda': 
            mask = self.process_mask(mask)
        

        if self.mode == 'val':
            np_img = np.array(img)
            img, mask = normalize(img, mask)
            return np_img, img, mask, id
