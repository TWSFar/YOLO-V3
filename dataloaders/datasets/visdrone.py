# -*- coding: UTF-8 -*-
import glob
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import xyxy2xywh
from dataloaders.transforms import *
import xml.etree.ElementTree as ET
IMG_ROOT = 'images'
ANNO_ROOT = 'annotations'


class VisdroneDataset(Dataset):  # for training
    classes = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

    def __init__(self, root_name='data', img_size=608, hyp=None, train=False):
        # load image path
        self._root = root_name
        self._anno_path = os.path.join(self._root, ANNO_ROOT, '{}.txt')
        self._image_path = os.path.join(self._root, IMG_ROOT, '{}.jpg')
        self._items = self._load_items()
        self.img_files = [self._image_path.format(x) for x in self._items]
        self.label_files = [self._anno_path.format(x) for x in self._items]

        # information of train
        self.img_num = len(self.img_files)
        assert self.img_num > 0, 'No images found in %s' % self._image_path
        self.img_size = img_size
        self.train = train
        self.hyp = hyp

        # classes
        self.num_classes = len(self.classes)
        self.index_map = dict(zip(self.classes, range(self.num_classes)))

    def _load_items(self):
        """Load individual image indices from splits."""
        ids = []
        root = self._root
        files = os.listdir(os.path.join(root, IMG_ROOT))
        ids += [line.strip()[:-4] for line in files]
        return ids

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # read img and label
        img = cv2.imread(self.img_files[index])  # BGR
        assert img is not None, 'File Not Found ' + self.img_files[index]
        h, w = img.shape[:2]
        hyp = self.hyp

        labels = self._load_visdrone_annotation(self.label_files[index])

        # print(labels)
        if self.train:
            # hsv
            img = augment_hsv(img, fraction=0.5)
            # # random crop
            labels, crop = random_crop_with_constraints(labels, (w, h))
            img = img[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :].copy()
            # # pad and resize
            img, labels = letterbox(img, labels, height=self.img_size, train=self.train)
            # Augment image and labels
            img, labels = random_affine(img, labels,
                                        degrees=hyp['degrees'],
                                        translate=hyp['translate'],
                                        scale=hyp['scale'],
                                        shear=hyp['shear'])
            # random left-right flip
            img, labels = random_flip(img, labels, 0.5)
            # color distort
            # img = random_color_distort(img)
        else:
            # pad and resize
            img, labels = letterbox(img, labels, height=self.img_size, train=self.train)

        nL = len(labels)
        # show_image(img, labels)

        if nL > 0:
            # convert xyxy to xywh 
            labels = np.clip(labels, 0, self.img_size - 1) # 这里可能存在bug, 当类的数量大于输入图像的输入大小时, 就炸了???
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy())

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = torch.from_numpy(img).float()
        labels_out = labels_out.float()
        shape = np.array([h, w], dtype=np.float32)
        return (img, labels_out, self.img_files[index], shape)

    @staticmethod
    def collate_fn(batch):
        img, label, hw, path = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), hw, path

    def __len__(self):
        return self.img_num  # number of image

    def validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _load_visdrone_annotation(self, file):
        with open(file, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)
        box_all = []
        bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.float64)
        for bbox in bboxes:
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bbox[5] -= 1
            box_all.append(bbox[[5, 0, 1, 2, 3]].tolist())  # (label(0-9),x1,y1,x2,y2)

        return np.array(box_all)


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '-')
    plt.show()
