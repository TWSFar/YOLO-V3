import os
import time
import os.path as osp
import shutil
import torch
import glob


class Saver(object):

    def __init__(self, opt, hyp, mode='train'):
        self.opt = opt
        self.hyp = hyp
        self.directory = osp.join('run', opt.dataset)
        experiment_name = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = osp.join(self.directory, experiment_name + '_' + mode)
        self.logfile = osp.join(self.experiment_dir, 'experiment.log')
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        log = {}
        log.update(self.opt._state_dict())
        log.update(self.hyp)
        for key, val in log.items():
            line = key + ': ' + str(val)
            self.save_experiment_log(line)

    def save_checkpoint(self, state, is_best, filename='checkpoint.path.tar'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(osp.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('epoch {}: {}'.format(state['epoch'], best_pred))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))

    def save_experiment_log(self, line):
        with open(self.logfile, 'a') as f:
            f.write(line + '\n')

    def save_eval_result(self, stats):
        with open(os.path.join(self.experiment_dir, 'result.txt'), 'a') as f:
            f.writelines(stats + '\n')
