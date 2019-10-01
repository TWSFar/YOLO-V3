import os
import os.path as osp


annos = os.listdir("/home/twsf/data/VOC2012/Annotations")
val = "/home/twsf/data/VOC2012/ImageSets/Main/val.txt"
vals = []

with open(val, 'r') as f:
    lines = f.readlines()
    for line in lines:
        vals.append(line.strip())
aug_train = []


for anno in annos:
    idx = anno.strip()[:-4]
    if idx not in vals:
        aug_train.append(idx.strip())
pass

aug_train_path = "/home/twsf/data/VOC2012/ImageSets/Main/aug_train.txt"

with open(aug_train_path, 'w') as f:
    for line in aug_train:
        f.write(line+'\n')
# for line in range
