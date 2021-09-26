# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

matplotlib.pyplot.switch_backend('Agg')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_graph(folder):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    names = ['timestamp', 'epoch', 'loss_t', 'loss_v', 'dice', 'mean', 'lr']
    info = pd.read_csv(
        os.path.join(folder, 'progress.csv'), header=None,
        index_col=False, names=names)

    name = folder + '/Progress.png'
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(info['epoch'], info['loss_t'], label='Train')
    ax1.plot(info['epoch'], info['loss_v'], label='Val')
    ax1.set_ylim(top=2, bottom=0)
    ax1.grid()
    ax1.set(title='Prueba ' + folder)

    dice = [i.strip('[').strip(']').split(' ') for i in info['dice']]
    dice = [list(filter(('').__ne__, i)) for i in dice]
    dice = np.asarray(dice, dtype=float)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Dice')
    if dice.shape[1] > 1:
        for i in range(dice.shape[1]):
            ax2.plot(info['epoch'], dice[:, i], linewidth=0.75,
                     color=colors[i + 3], label='Label ' + str(i))
    ax2.plot(info['epoch'], info['mean'], label='Mean', color=colors[2])
    ax2.legend()
    ax2.set_ylim(top=1, bottom=0)
    ax2.yaxis.grid(linestyle=(0, (1, 10)), linewidth=0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(name, dpi=300)

    plt.close('all')


def save_epoch(state, mean, save_path, out_file, checkpoint, is_best):
    out_file.write('{},{},{},{},{},{},{}\n'.format(
        datetime.now(), state['epoch'], state['loss'][0], state['loss'][1],
        state['dice'], mean, state['lr']))
    out_file.flush()

    if checkpoint:
        name = 'epoch_' + str(state['epoch']) + '.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('Checkpoint saved:', name)

    if is_best:
        name = 'best_dice.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('New best model saved')

    name = 'checkpoint.pth.tar'
    torch.save(state, os.path.join(save_path, name))
    save_graph(save_path)


def one_hot(gt, categories):
    # Check the new function in PyTorch!!!
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 4, 1, 2, 3).contiguous()
    return gt


# ================= Keep stats =================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluator(object):
    '''Used to calculate the metrics'''
    def __init__(self, num_class):
        self.num_class = num_class
        self.conf = np.zeros((self.num_class,) * 2)  # Confusion matrix

    def Pixel_Accuracy(self):
        Acc = np.diag(self.conf).sum() / self.conf.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.conf) / self.conf.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.conf) / (
            np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0) -
            np.diag(self.conf))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.conf, axis=1) / np.sum(self.conf)
        iu = np.diag(self.conf) / (
            np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0) -
            np.diag(self.conf))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Dice_Score(self):
        MDice = (2 * np.diag(self.conf)) / (
            np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0))
        return MDice[1:]  # Only foreground

    def Frequency_Weighted_Dice_Score(self):
        freq = np.sum(self.conf, axis=1) / np.sum(self.conf)
        dice = (2 * np.diag(self.conf)) / (
            np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0))

        FWDice = (freq[freq > 0] * dice[freq > 0]).sum()
        return FWDice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        conf = count.reshape(self.num_class, self.num_class)
        return conf

    def add_batch(self, pre_image, gt_image):
        assert gt_image.shape == pre_image.shape
        self.conf += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.conf = np.zeros((self.num_class,) * 2)
