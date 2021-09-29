# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import libs.utilities.utils as utils
import libs.dataloader.helpers as helpers

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F


def train(args, info, model, loader, noise_data, optimizer, criterion, scaler,
          rank):
    model.train()
    loader.dataset.change_epoch()
    epoch_loss = utils.AverageMeter()
    batch_loss = utils.AverageMeter()

    iterations = args.adv_iters if args.AT else 1
    print_freq = max(1, len(loader) // 2)
    eps = args.eps / 255.
    for batch_idx, sample in enumerate(loader):
        data = sample['data'].float().to(rank)

        # Rescale (important for Free AT) ¡rescale the eps!
        b_min = torch.amin(data, [2, 3, 4], keepdim=True)
        b_max = torch.amax(data, [2, 3, 4], keepdim=True)
        b_eps = (b_max - b_min) * eps
        # data = (data - b_min) / (b_max - b_min + 1e-5)

        target = sample['target'].squeeze_(1).long().to(rank)
        for _ in range(iterations):
            optimizer.zero_grad()
            in_data = data
            if args.AT:
                delta = noise_data[0:data.size(0)].to(rank)
                delta.requires_grad = True
                in_data = torch.maximum(
                    torch.minimum(data + delta, b_max), b_min)

            with amp.autocast():
                out = model(in_data)
                loss = criterion(out, target)

            scaler.scale(loss).backward()

            if args.AT:
                # Update the adversarial noise
                grad = delta.grad.detach().cpu()
                noise_data[0:data.size(0)] += (b_eps * torch.sign(grad)).data
                noise_data = torch.clamp(noise_data, -b_eps, b_eps)

            scaler.step(optimizer)
            scaler.update()
            batch_loss.update(loss.item())
            epoch_loss.update(loss.item())

        if batch_loss.count % print_freq == 0:
            if rank == 0:
                text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(text.format(
                    time.strftime("%H:%M:%S"), (batch_idx + 1),
                    (len(loader)), 100. * (batch_idx + 1) / (len(loader)),
                    batch_loss.avg))
            batch_loss.reset()
    if rank == 0:
        print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg, noise_data


def val(args, model, loader, criterion, metrics, rank):
    model.eval()
    metrics.reset()
    epoch_loss = utils.AverageMeter()

    for _, sample in enumerate(loader):
        data = sample['data'].float().to(rank)
        target = sample['target'].squeeze_(1).long()

        with torch.no_grad():
            out = model(data)
        loss = criterion(out, target.to(rank))

        prediction = F.softmax(out, dim=1)
        prediction = torch.argmax(prediction, dim=1).cpu().numpy()
        metrics.add_batch(target.numpy(), prediction)

        epoch_loss.update(loss.item(), n=target.shape[0])
    dice = metrics.Dice_Score()
    if rank == 0:
        print('--- Val: \tLoss: {:.6f} \tDice fg: {} ---'.format(
            epoch_loss.avg, dice))
    return epoch_loss.avg, dice


def test(info, model, loader, images_path, test_file, rank, world_size):
    '''
    The inference is done by uniformly extracting patches of the images.
    The patches migth overlap, so we perform a weigthed average based on
    the distance of each voxel to the center of their corresponding patch.
    '''
    patients = pd.read_csv(test_file)
    model.eval()

    # Add more weight to the central voxels
    w_patch = np.zeros(info['val_size'])
    sigmas = np.asarray(info['val_size']) // 8
    center = torch.Tensor(info['val_size']) // 2
    w_patch[tuple(center.long())] = 1
    w_patch = gaussian_filter(w_patch, sigmas, 0, mode='constant', cval=0)
    w_patch = torch.Tensor(w_patch / w_patch.max()).to(rank).half()

    for idx in range(rank, len(patients), world_size):
        shape, name, affine, pad = loader.dataset.update(idx)
        prediction = torch.zeros((info['classes'],) + shape).to(rank).half()
        weights = torch.zeros(shape).to(rank).half()

        for sample in loader:
            data = sample['data'].float()  # .squeeze_(0)
            with torch.no_grad():
                output = model(data.to(rank))[0]
            # output = dataloader.test_data(output, False)
            output *= w_patch

            low = (sample['target'][0] - center).long()
            up = (sample['target'][0] + center).long()
            prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output
            weights[low[0]:up[0], low[1]:up[1], low[2]:up[2]] += w_patch

        prediction /= weights
        prediction = F.softmax(prediction, dim=0)
        prediction = torch.argmax(prediction, dim=0).cpu()
        if pad is not None:
            prediction = prediction[pad[0][0]:shape[0] - pad[0][1],
                                    pad[1][0]:shape[1] - pad[1][1],
                                    pad[2][0]:shape[2] - pad[2][1]]

        helpers.save_image(
            prediction, os.path.join(images_path, name), affine)
        print('Prediction {} saved'.format(name))
