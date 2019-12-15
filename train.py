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


class Trainer(object):
    def __init__(self):
        init_seeds(opt.seed)
        self.start_epoch = 0
        self.best_pred = 0.
        self.cutoff = -1  # backbone reaches to cutoff layer

        # Define Saver
        self.saver = Saver(opt, hyp)

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        if 'pw' not in opt.arc:  # remove BCELoss positive weights
            hyp['cls_pw'] = 1.
            hyp['obj_pw'] = 1.

        if opt.multi_scale:
            self.img_size_min = round(opt.img_size / 32 / 1.5) + 1
            self.img_size_max = round(opt.img_size / 32 * 1.5) - 1
            self.img_size = self.img_size_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (self.img_size_min*32, self.img_size_max*32))
        else:
            self.img_size = opt.img_size

        # Define Dataloader
        self.train_dataset, self.train_loader = make_data_loader(opt, hyp, train=True)
        self.tnb = len(self.train_loader)
        self.num_classes = self.train_dataset.num_classes
        self.val_dataset, self.val_loader = make_data_loader(opt, hyp, train=False)
        self.vnb = len(self.val_loader)

        # Initialize model
        self.model = Darknet(opt.cfg, self.img_size, opt.arc).to(opt.device)
        self.model.nc = self.num_classes  # attach number of classes to model
        self.model.arc = opt.arc  # attach yolo architecture
        self.model.hyp = hyp  # attach hyperparameters to model

        # Optimizer
        if opt.adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=hyp['lr'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=hyp['lr'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

        # load weight
        if opt.resume:  # pytorch format
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        elif len(opt.weights) > 0:  # darknet format
            self.cutoff = load_darknet_weights(self.model, opt.weights)

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        # Initialize distributed training
        if len(opt.gpu_id) > 1:
            print("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_id)

        # transfer learning edge (yolo) layers
        if opt.transfer or opt.prebias:
            nf = int(self.model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

            for p in self.optimizer.param_groups:
                # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                p['lr'] *= 100
                if p.get('momentum') is not None:  # for SGD but not Adam
                    p['momentum'] *= 0.9

            for p in self.model.parameters():
                if opt.prebias and p.numel() == nf:  # train (yolo biases)
                    p.requires_grad = True
                elif opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                    p.requires_grad = True
                else:  # freeze layer
                    p.requires_grad = False

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  milestones=[round(opt.epochs * x) for x in opt.steps],
                                                  gamma=opt.gamma)
        self.scheduler.last_epoch = self.start_epoch - 1

        # Time
        self.timer = Timer(opt.epochs, self.tnb, self.vnb)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def training(self, epoch):
        self.model.train()
        mloss = torch.zeros(5).to(opt.device)  # mean losses
        for iter_num, (imgs, targets, paths, _) in enumerate(self.train_loader):
            temp_time = time.time()
            global_step = iter_num + self.tnb * epoch + 1
            imgs = imgs.to(opt.device)
            targets = targets.to(opt.device)

            # Multi-Scale training
            if opt.multi_scale:
                if global_step % opt.print_freq == 0:  #  adjust (67% - 150%) every print_freq batches
                    self.img_size = random.randrange(self.img_size_min, self.img_size_max + 1) * 32
                sf = self.img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            pred = self.model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, self.model)
            assert torch.isfinite(loss), "WARNING: non-finite loss, ending training"      

            # Scale loss by nominal batch_size of 64
            loss *= opt.batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (iter_num + 1) % opt.accumulate == 0 or (iter_num + 1) == self.tnb:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # visulization
            titles = ['xy', 'wh', 'obj', 'cls', 'total']
            for title, loss_item in zip(titles, loss_items):
                self.writer.add_scalar('train/{}_loss'.format(title), loss_item.cpu().item(), global_step)
 
            batch_time = time.time() - temp_time
            eta = self.timer.eta(global_step, batch_time)
            self.step_time.append(batch_time)
            mloss = (mloss * iter_num + loss_items) / (iter_num + 1)  # update mean losses
            if global_step % opt.print_freq == 0:
                printline = ("Epoch: [{}][{}/{}]  "
                            "lr: {}, eta: {}, time: {:1.3f}, "
                            "loss(xy: {:1.5f}, wh: {:1.5f}, "
                            "obj: {:1.5f}, cls: {:1.5f}"
                            "total loss: {:1.5f}), img_size: {}").format(
                            epoch, iter_num + 1, self.tnb,
                            self.optimizer.param_groups[0]['lr'],
                            eta, np.sum(self.step_time),
                            *mloss, self.img_size)
                print(printline)
                self.saver.save_experiment_log(printline)

        # Update scheduler
        self.scheduler.step()

    def validate(self, epoch):
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
                global_step = batch_i + self.vnb * epoch
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

            # visualize
            titles = ['Precision', 'Recall', 'mAP', 'F1']
            result = [mp, mr, map, mf1]
            for xi, title in zip(result, titles):
                self.writer.add_scalar('val/{}'.format(title), xi, epoch)

            # Print and Write results
            title = ('%10s' * 7) % ('epoch: [{}]'.format(epoch), 'Class', 'Images', 'P', 'R', 'mAP', 'F1')
            print(title)
            self.saver.save_eval_result(stats=title)
            printline = '%20s' + '%10.3g' * 5
            pf = printline % ('all', seen, mp, mr, map, mf1) # print format
            print(pf)
            self.saver.save_eval_result(stats=pf)
            if self.num_classes > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    pf = printline % (self.val_dataset.classes[c], seen, p[i], r[i], ap[i], f1[i])
                    print(pf)
                    self.saver.save_eval_result(stats=pf)
            return map


def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()
    model_info(trainer.model, report='summary')
    print('Num training images: {}'.format(trainer.train_dataset.img_num))

    for epoch in range(trainer.start_epoch, opt.epochs):
        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in trainer.model.named_parameters():
                if int(name.split('.')[1]) < trainer.cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        trainer.training(epoch)

        # validate
        val_time = time.time()
        map = trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        # Update best mAP
        is_best = map > trainer.best_pred
        trainer.best_pred = max(map, trainer.best_pred)
        if (epoch % opt.saver_freq == 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    train()
