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
from .transforms import *
import xml.etree.ElementTree as ET

class LoadImages():  # for inference
    def __init__(self, path, img_size=416):
        self.height = img_size
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'File Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, _, = letterbox(img0, None, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, mode='test')

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training
    def __init__(self, root='data', img_size=608, batch_size=8, hyp=None,
                 classes=[], mode='train'):

        # load image path
        self._root = root
        self._items = self._load_items(mode)
        self._anno_path = os.path.join(root, 'Annotations', '{}.xml')
        self._image_path = os.path.join(root, 'JPEGImages', '{}.jpg')
        self.img_files = [self._image_path.format(x) for x in self._items]
        self.label_files = [self._anno_path.format(x) for x in self._items]

        # information of train
        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path
        self.nF = n     # number of image files
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.mode = mode
        self.hyp = hyp

        # classes
        self.classes = classes
        self.index_map = dict(zip(self.classes, range(self.num_class)))


    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)


    def _load_items(self, mode):
        """Load individual image indices from splits."""
        ids = []
        root = self._root
        # lf = os.path.join(root, 'ImageSets', 'Main', mode + '.txt')
        lf = os.path.join(root, 'ImageSets', 'Main', mode + '.txt')
        with open(lf, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
        return ids


    def __getitem__(self, index):           
        assert index <= len(self), 'index range error'
        # read img and label
        img = cv2.imread(self.img_files[index])  # BGR
        assert img is not None, 'File Not Found ' + self.img_files[index]
        h, w = img.shape[:2]
        hyp = self.hyp

        labels = self._load_pascal_annotation(self.label_files[index], self.index_map)
        try:
            if labels.shape[0]:
                assert labels.shape[1] == 5, '> 5 label columns: %s' % file
                assert (labels >= 0).all(), 'negative labels: %s' % file
                assert (labels[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
        except:
            pass  # print('Warning: missing labels for %s' % self.img_files[i])  # missing label file
        assert len(np.concatenate(labels, 0)) > 0, 'No labels found. Incorrect label paths provided.'

        #print(labels)
        if self.mode in ['train', 'trainval']:
            # hsv
            img = augment_hsv(img, fraction=0.5)
            # # random crop
            labels, crop = random_crop_with_constraints(labels, (w, h))
            img = img[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :].copy()
            # # pad and resize
            img, labels = letterbox(img, labels, height=self.img_size, mode=self.mode)
            # Augment image and labels
            img, labels = random_affine(img, labels,
                                        degrees=hyp['degrees'],
                                        translate=hyp['translate'],
                                        scale=hyp['scale'],
                                        shear=hyp['shear'])
            # random left-right flip
            img, labels = random_flip(img, labels, 0.5)
            # color distort
            img = random_color_distort(img)
        elif self.mode in ['test', 'val']:
            # pad and resize
            img, labels = letterbox(img, labels, height=self.img_size, mode=self.mode)

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
        # img /= 255.0

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = torch.from_numpy(img).float()
        labels_out = labels_out.float()
        shape = np.array([h,w], dtype=np.float32)
        return (img, labels_out, self.img_files[index], shape)
    

    @staticmethod
    def collate_fn(batch):
        img, label, hw, path = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), hw, path

    def __len__(self):
        return self.nF  # number of batches


    def validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


    def _load_pascal_annotation(self, file, index_map):
        tree = ET.parse(file)
        size=tree.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        objs = tree.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 5), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = obj.find('name').text
            try:
                boxes[ix, 0] = index_map[cls]
            except:
                continue
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            try:
                self.validate_label(x1, y1, x2, y2, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))

            boxes[ix, 1:] = [x1, y1, x2, y2]

        return boxes


def convert_tif2bmp(p='../xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))
        cv2.imwrite(f.replace('.tif', '.bmp'), cv2.imread(f))
        os.system('rm -rf ' + f)


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '-')
    plt.show()
