import os
import time
import numpy as np
from pprint import pprint
from utils.devices import select_device

user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    # root_path = "/home/twsf/data/Visdrone"
    root_path = "/home/twsf/work/YOLO-V3/data/VisDrone"
    img_size = 960

    # model
    resume = False
    pre = "/home/twsf/work/YOLO-V3/run/visdrone/20191205_203623/model_best.pth.tar"
    cfg = 'cfg/yolov3-spp-visdrone.cfg'
    weights = '/home/twsf/.cache/torch/checkpoints/darknet53.conv.74'
    transfer = False
    prebias = False

    # train
    img_weights = False
    multi_scale = True
    rect = False
    batch_size = 2
    accumulate = 1
    epochs = 70
    workers = 1

    # param for optimizer
    adam = False
    steps = [0.8, 0.9]
    gamma = 0.1

    # eval
    # parameters
    iou_thres = 0.5
    pst_thd = 0.2
    nms_thd = 0.5
    # 'OR' (default), 'AND', 'MERGE' (experimental)
    nms_style = 'OR'

    # loss
    arc = 'default'
    giou_loss = False

    # visual
    visualize = True
    print_freq = 10
    plot_freq = 50  # every n batch plot
    saver_freq = 1

    seed = int(time.time())

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.device, self.gpu_id = select_device()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()