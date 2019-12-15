import math
import time
import argparse
import collections

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from config.visdrone import opt
from models.darknet import Darknet, load_darknet_weights
from dataloaders import make_data_loader
from utils.utils import *
from utils.timer import Timer
from utils.saver import Saver
from utils.visualization import TensorboardSummary

import multiprocessing
multiprocessing.set_start_method('spawn', True)
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

hyp = {'giou': 1.582,  # giou loss gain
       'xy': 0.20,  # xy loss gain
       'wh': 0.10,  # wh loss gain
       'cls': 0.035,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'conf': 1.61,  # obj loss gain (*=80 for uBCE with 80 classes)
       'conf_bpw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.3,  # iou training threshold
       'lr': 0.0005,  # learning rate
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.05,  # focal loss gamma
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)


class Evaluate(object):
    def __init__(self):
        init_seeds(opt.seed)
        self.best_pred = 0.
        self.cutoff = -1  # backbone reaches to cutoff layer

        # Define Saver
        self.saver = Saver(opt, hyp, mode='val')

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        if 'pw' not in opt.arc:  # remove BCELoss positive weights
            hyp['cls_pw'] = 1.
            hyp['obj_pw'] = 1.

        self.img_size = opt.img_size

        # Define Dataloader
        self.val_dataset, self.val_loader = make_data_loader(opt, hyp, train=False)
        self.num_classes = self.val_dataset.num_classes
        self.vnb = len(self.val_loader)

        # Initialize model
        self.model = Darknet(opt.cfg, self.img_size, opt.arc).to(opt.device)
        self.model.nc = self.num_classes  # attach number of classes to model
        self.model.arc = opt.arc  # attach yolo architecture
        self.model.hyp = hyp  # attach hyperparameters to model

        # load weight
        if os.path.isfile(opt.pre):
            print("=> loading checkpoint '{}'".format(opt.pre))
            checkpoint = torch.load(opt.pre)
            self.epoch = checkpoint['epoch']
            self.best_pred = checkpoint['best_pred']
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(opt.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.pre))

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        # Initialize distributed training
        if len(opt.gpu_id) > 1:
            print("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_id)

    def validate(self):
        seen = 0
        self.model.eval()
        with torch.no_grad():
            p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
            # loss = torch.zeros(3)
            jdict, stats, ap, ap_class = [], [], [], []
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(self.val_loader)):
                targets = targets.to(opt.device)
                imgs = imgs.to(opt.device)
                _, _, height, width = imgs.shape  # batch size, channels, height, width

                # pred
                inf_out, train_out = self.model(imgs)  # inference and training outputs

                # # Compute loss
                # if hasattr(self.model, 'hyp'):  # if model has loss hyperparameters
                #     loss += compute_loss(train_out, targets, self.model)[1][:3].cpu()  # GIoU, obj, cls

                # Run NMS: output as (x1y1x2y2, obj_conf, class_conf, class_pred)
                output = non_max_suppression(
                    inf_out,
                    conf_thres=opt.pst_thd, nms_thres=opt.nms_thd,
                    nms_style=opt.nms_style)

                # Visualization
                global_step = batch_i + self.vnb
                if global_step % opt.plot_freq == 0:
                    predict = []
                    for i in range(len(output)):
                        temp = output[i]
                        if temp is None:
                            predict.append(torch.zeros(0, 5))
                        else:
                            # (x1, y1, x2, y2, class_pred, obj_conf)
                            predict.append(temp[:, [0, 1, 2, 3, 6, 4]])
                    self.summary.visualize_image(
                        self.writer,
                        imgs, targets, predict,
                        self.val_dataset.classes,
                        global_step)

                # Statistics per image
                for si, pred in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]  # si: i'th image
                    nl = len(labels)  # number of object
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if nl:
                            stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Assign all predictions as incorrect
                    correct = [0] * len(pred)
                    if nl:
                        detected = []
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5])
                        tbox[:, [0, 2]] *= width
                        tbox[:, [1, 3]] *= height

                        # Search for correct predictions
                        for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                            # Break if all targets already located in image
                            if len(detected) == nl:
                                break

                            # Continue if predicted class not among image classes
                            if pcls.item() not in tcls:
                                continue

                            # Best iou, index between pred and targets
                            m = (pcls == tcls_tensor).nonzero().view(-1)
                            iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                            # If iou > threshold and class is correct mark as correct
                            if iou > opt.iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                                correct[i] = 1
                                detected.append(m[bi])

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

            # Compute statistics
            stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
            if len(stats):
                p, r, ap, f1, ap_class = ap_per_class(*stats)
                mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

            # Print and Write results
            title = ('%10s' * 6) % ('Class', 'Images', 'P', 'R', 'mAP', 'F1')
            print(title)
            self.saver.save_eval_result(stats=title)
            printline = '%10s' + '%10.3g' * 5
            pf = printline % ('all', seen, mp, mr, map, mf1) # print format
            print(pf)
            self.saver.save_eval_result(stats=pf)
            if self.num_classes > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    pf = printline % (self.val_dataset.classes[c], seen, p[i], r[i], ap[i], f1[i])
                    print(pf)
                    self.saver.save_eval_result(stats=pf)


def eval(**kwargs):
    opt._parse(kwargs)
    evaluater = Evaluate()
    model_info(evaluater.model, report='summary')
    print('Num validate images: {}'.format(evaluater.val_dataset.img_num))

    evaluater.validate()


if __name__ == '__main__':
    eval()
