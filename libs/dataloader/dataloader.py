# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

import libs.dataloader.helpers as helpers

import torch.nn as nn
import torch.nn.functional as F


class Medical_data(Dataset):
    def __init__(self, train, csv_file, root_dir, patch_size, im_path=None,
                 val=False, pgd=False):
        super(Medical_data, self).__init__()
        self.csv_file = csv_file
        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.val = val
        self.patch_size = np.asarray(patch_size)
        self.fg = 0
        self.pgd = pgd
        self.images_path = im_path
        self.folder = ''

    def __len__(self):
        if self.train or self.pgd:
            return len(self.filenames)
        else:
            return len(self.voxel)

    def __getitem__(self, idx):
        if self.train or self.pgd:
            patient = self.filenames.iloc[int(idx)]
            image, label, affine = helpers.load_image(
                patient, self.root_dir, self.train)
            im_shape, multimodal = helpers.image_shape(image)

            # If the image is smaller than the patch_size in any dimension, we
            # have to pad it to extract a patch
            if any(im_shape <= self.patch_size):
                dif = (self.patch_size - im_shape) // 2 + 3
                pad = np.maximum(dif, [0, 0, 0])
                pad_lb = tuple(zip(pad, pad))
                label = np.pad(label, pad_lb, 'reflect')

                if multimodal:
                    pad_im = [0] + pad.tolist()
                else:
                    pad_im = pad
                pad_im = tuple(zip(pad_im, pad_im))
                image = np.pad(image, pad_im, 'reflect')

            if self.val:
                voxel = np.asarray(label.shape) // 2
            else:
                fg = (idx + self.fg) % 2 == 0
                voxel = helpers.train_voxels(image, self.patch_size, label, fg)

            if self.train:
                # Patch extraction
                patches = helpers.extract_patch(image, voxel, self.patch_size)
                label = helpers.extract_patch(label, voxel, self.patch_size)
                patches = patches.astype(float)
                info = 0
            elif self.pgd:
                patches = image
                if len(patches.shape) == 3:
                    patches = np.expand_dims(patches, 0)
                label = np.expand_dims(label, 0)
                info = [patient[0][11:], affine]
        else:
            patches = helpers.extract_patch(
                self.image, self.voxel[idx], self.patch_size)
            label = torch.Tensor(self.voxel[idx])
            # patches = helpers.test_data(patches)
            patches = torch.from_numpy(patches)
            info = 0

        return {'data': patches, 'target': label, 'info': info}

    def change_epoch(self):
        self.fg = 1 - self.fg

    def update(self, im_idx):
        # This is only for testing
        patient = self.filenames.iloc[im_idx]
        name = patient[0][11:]  # ./imagesTr/XXXX.nii.gz (or imagesTs)
        print('Loading data of patient {} ---> {}'.format(
            name, time.strftime("%H:%M:%S")))

        image, _, affine = helpers.load_image(
            patient, self.root_dir, self.train)
        self.image, pad = helpers.verify_size(image, self.patch_size)
        im_shape, multimodal = helpers.image_shape(self.image)
        if multimodal and pad is not None:
            pad = pad[1:]

        self.voxel = helpers.test_voxels(self.patch_size, im_shape)
        return im_shape, name, affine, pad

    def update_pgd(self, im_idx):
        # This is only for testing the adversarial robustness
        patient = self.filenames.iloc[im_idx]
        name = patient[0][11:]  # ./imagesTr/XXXX.nii.gz (or imagesTs)

        image, label, affine = helpers.load_image(
            patient, self.root_dir, self.train)
        image = image.astype(np.float32)
        image, _ = helpers.verify_size(image, self.patch_size)
        label, _ = helpers.verify_size(label, self.patch_size)
        assert image.shape[-3:] == label.shape[-3:]
        im_shape, _ = helpers.image_shape(image)

        # voxel = np.asarray(label.shape) // 2
        voxel = helpers.val_voxels(im_shape, self.patch_size, label)
        image = helpers.extract_patch(image, voxel, self.patch_size)
        image = torch.from_numpy(image).unsqueeze(0)  # Batch dimension
        label = helpers.extract_patch(label, voxel, self.patch_size)
        return image, torch.from_numpy(label), name, affine
