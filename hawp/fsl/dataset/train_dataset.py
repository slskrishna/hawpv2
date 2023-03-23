import os
import math
import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from random import randint as ri
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy


class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, augmentation=4):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
        self.augmentation = augmentation
        print(len(self.annotations))
        mask_ann_path = "../hawp/data/wireframe/mask_data.json"
        assert osp.isfile(mask_ann_path), "Mask annotations are missing"
        with open(mask_ann_path, "r") as f:
            self.mask_annotations = json.load(f)
        self.count = 0
        self.images = 0
        occ_folder = "../hawp/data/Stock images/"
        self.occ_files = [occ_folder + file for file in os.listdir(occ_folder) if "DS_" not in file]
        self.occ_length = len(self.occ_files)
        self.outer_contour = json.load(open("../hawp/data/wireframe/train_outer_contour.json"))

    def __getitem__(self, idx_):
        idx = idx_ % len(self.annotations)
        reminder = idx_ // len(self.annotations)

        ann = copy.deepcopy(self.annotations[idx])
        if len(ann['edges_negative']) == 0:
            ann['edges_negative'] = [[0, 0]]
        image = io.imread(osp.join(self.root, ann['filename'])).astype(float)  # [:,:,:3]
        if len(image.shape) == 2:
            image = np.concatenate([image[..., None], image[..., None], image[..., None]], axis=-1)
        else:
            image = image[:, :, :3]

        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.long],
                           ['edges_negative', np.long]):
            ann[key] = np.array(ann[key], dtype=_type)
        width = ann['width']
        height = ann['height']
        if (random.randint(1, 10) <= 10):
            mask = cv2.imread(self.occ_files[ri(0, self.occ_length - 1)])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            sx, sy, hm, wm = ri(165, 286), ri(165, 286), len(mask), len(mask[0])
            lines = self.outer_contour.get(ann['filename'], [])
            ratio = math.sqrt(random.choice([4900, 6400, 8100, 10000, 12100, 14400, 16900]) / (hm * wm))
            hm, wm = int(ratio * hm), int(wm * ratio)
            mask = cv2.resize(mask, (wm, hm))
            if lines and False:
                idx = random.randint(0, len(lines) - 1)
                [x1, y1], [x2, y2] = lines[idx], lines[(idx + 1) % len(lines)]
                ratio = random.randint(1, 99) / 100
                sx, sy = int(x1 + ratio * (x2 - x1)), int(y1 + ratio * (y2 - y1))
                sx, sy = max(0, sx - wm // 2), max(0, sy - hm // 2)

            mask_of_mask = (mask[:, :, 0] < 160) | (mask[:, :, 1] < 160) | (mask[:, :, 2] < 160)
            try:
                mask = np.clip((mask * 0.4) + 30, 0, 255)
                image[sy: sy + hm, sx: sx + wm][mask_of_mask] = mask[mask_of_mask]
            except:
                pass

        # Try image blurring or sharpening
        if (ri(1, 10) <= 4):
            image = cv2.GaussianBlur(image, (2 * ri(1, 3) + 1, 2 * ri(1, 3) + 1), 0)

        # Try contrast or brightness augmentation
        if (ri(1, 10) <= 6):
            brightness = ri(0, 50)
            contrast = ri(7, 13) / 10
            image = np.clip((image * contrast) + brightness, 0, 255)

        reminder = random.randint(0, 6)
        ann['reminder'] = reminder
        if reminder == 1:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
        elif reminder == 2:
            image = image[::-1, :, :]
            mask = mask[::-1, :]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
        elif reminder == 3:
            image = image[::-1, ::-1, :]
            mask = mask[::-1, ::-1]
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
        elif reminder >= 4:
            angle = random.randint(1, 3)
            M = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * 90, 1)
            image = cv2.warpAffine(image, M, (height, width))
            ann['junctions'] -= 255.5
            ann['junctions'] = np.dot(ann['junctions'],
                                      cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * -90, 1))[:,
                               :2]
            ann['junctions'] += 255.5
            ann['junctions'] = ann['junctions'].astype('float32')
        else:
            pass
        if self.transform is not None:
            return self.transform(image, ann)

        return image, ann

    def __len__(self):
        return len(self.annotations) * self.augmentation


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
