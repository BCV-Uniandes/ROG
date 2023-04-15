# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import nibabel as nib
from scipy import ndimage

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import GammaTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform


# It's faster to include the data augmentation in the dataloader's collate_fn
class collate(object):
    def __init__(self, size):
        rot_angle = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        # Rotation (x, y, z), scale
        spatial = [
            SpatialTransform(
                size, label_key='target', do_elastic_deform=False,
                do_rotation=True, p_rot_per_sample=0.5, angle_x=rot_angle,
                angle_y=rot_angle, angle_z=rot_angle, do_scale=True,
                p_scale_per_sample=0.5, scale=(0.85, 1.15),
                border_mode_data='constant', border_cval_data=0,
                order_data=3, border_mode_seg='constant',
                border_cval_seg=-1, order_seg=0, random_crop=False),
            MirrorTransform(label_key='target', axes=(0, 1, 2)),
            RemoveLabelTransform(output_key='target', input_key='target',
                                 replace_with=0, remove_label=-1)]

        # self.transformed = Compose(spatial + [NumpyToTensor()])
        self.transformed = Compose(spatial + [
            GammaTransform((0.7, 1.5), invert_image=False,
                            per_channel=False, retain_stats=True,
                            p_per_sample=0.2),
            ContrastAugmentationTransform(
                p_per_sample=0.2, preserve_range=True, per_channel=False),
            BrightnessTransform(0, 1, p_per_sample=0.2, per_channel=False),
            NumpyToTensor()])

    def __call__(self, batch):
        elem = batch[0]
        batch = {key: np.stack([d[key] for d in batch]) for key in elem}
        return self.transformed(**batch)


class collate_val(object):
    def __init__(self, size):
        self.cropped = Compose([
            CenterCropTransform(size, label_key='target'),
            NumpyToTensor(keys=['data', 'target'])])

    def __call__(self, batch):
        elem = batch[0]
        batch = {key: np.stack([d[key] for d in batch]) for key in elem}
        return self.cropped(**batch)


# I'm not using this yet (CHECK DATASETS WITH MULTIPLE MODALITIES)
def test_data(x, load=True):
    """ Test Time Augmentation """
    if load:
        axis = 0 if len(x.shape) == 4 else 1
        batch = [x]
        batch.append(np.flip(x, 3 + axis))
        batch.append(np.flip(x, 2 + axis))
        batch.append(np.flip(np.flip(x, 3 + axis), 2 + axis))
        batch.append(np.flip(x, 1 + axis))
        batch.append(np.flip(np.flip(x, 3 + axis), 1 + axis))
        batch.append(np.flip(np.flip(x, 2 + axis), 1 + axis))
        batch.append(np.flip(np.flip(
            np.flip(x, 2 + axis), 1 + axis), 3 + axis))
        return np.stack(batch, axis=0)
    else:
        result = x[0]
        result += np.flip(x[1], 3)
        result += np.flip(x[2], 2)
        result += np.flip(np.flip(x[3], 3), 2)
        result += np.flip(x[4], 1)
        result += np.flip(np.flip(x[5], 3), 1)
        result += np.flip(np.flip(x[6], 2), 1)
        result += np.flip(np.flip(np.flip(x[7], 2), 1), 3)
        return result / 8.


def load_image(patient, root_dir, train):
    gt = None
    im = nib.load(os.path.join(root_dir, patient['image']))
    affine = im.affine
    im = im.get_fdata()
    if len(im.shape) > 3:
        im = np.transpose(im, (3, 0, 1, 2))

    gt = nib.load(os.path.join(root_dir, patient['label'])).get_fdata()
    gt = gt.astype(np.int16)
    return im, gt, affine


def image_shape(im):
    im_shape = im.shape
    multimodal = False
    if len(im_shape) == 4:
        im_shape = im_shape[1:]
        multimodal = True
    return im_shape, multimodal


def test_voxels(patch_size, im_shape):
    """ Select the central voxels of the patches for testing """
    center = patch_size // 2
    dims = []
    for i, j in zip(im_shape, center):
        end = i - j
        num = np.ceil((end - j) / j)
        if num == 1:
            num += 1
        if num == 0:
            dims.append([i // 2])
            continue
        voxels = np.linspace(j, end, int(num))
        dims.append(voxels)
    voxels = list(itertools.product(*dims))
    return voxels


def val_voxels(im_shape, patch_size, label):
    low = patch_size // 2 - 1
    high = np.asarray(im_shape) - low
    pad = tuple(zip(low, low))
    mask = np.pad(np.ones(high - low), pad, 'constant')

    if (mask * label).sum() > 0:
        voxel = np.asarray(ndimage.measurements.center_of_mass(mask * label))
    else:
        voxel = np.asarray(ndimage.measurements.center_of_mass(label))
        nonzero = np.argwhere(mask == 1)
        distances = np.sqrt((nonzero[:, 0] - voxel[0]) ** 2 +
                            (nonzero[:, 1] - voxel[1]) ** 2 +
                            (nonzero[:, 2] - voxel[2]) ** 2)
        nearest_index = np.argmin(distances)
        voxel = nonzero[nearest_index]
    return voxel.astype(int)


def train_voxels(image, patch_size, label, foreground):
    """ Select the central voxels of the patches for testing """
    # Lower and upper bound to sample the central voxel (to avoid padding if
    # it is too close to the borders)
    im_shape, _ = image_shape(image)

    low = patch_size // 2
    high = np.asarray(im_shape) - low

    if foreground:
        # Force the center voxel to belong to a foreground category
        pad = tuple(zip(low, low))
        mask = np.pad(np.zeros(high - low), pad, 'constant',
                      constant_values=-1)

        np.copyto(mask, label, where=(mask == 0))
        fg = np.unique(mask)[2:]  # [ignore, bg, fg...]
        if fg.size > 0:
            cat = np.random.choice(fg)
            selected = np.argwhere(mask == cat)
            coords = selected[np.random.choice(len(selected))]
        else:
            x = np.random.randint(low[0], high[0])
            y = np.random.randint(low[1], high[1])
            z = np.random.randint(low[2], high[2])
            coords = (x, y, z)
    else:
        x = np.random.randint(low[0], high[0])
        y = np.random.randint(low[1], high[1])
        z = np.random.randint(low[2], high[2])
        coords = (x, y, z)
    return coords


def extract_patch(image, voxel, patch_size):
    im_shape, multimodal = image_shape(image)

    v1 = np.maximum(np.asarray(voxel) - patch_size // 2, 0)
    v1 = v1.astype(int)
    v2 = np.minimum(v1 + patch_size, im_shape)

    if multimodal:
        patch = image[:, v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
    else:
        patch = np.expand_dims(image[v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]], 0)
    patch, _ = verify_size(patch, patch_size)
    return patch


def verify_size(im, size):
    """ Verify if the patches have the correct size (if they are extracted
    from the borders they may be smaller) """
    im_shape, multimodal = image_shape(im)

    dif = np.asarray(size) - im_shape
    pad = None
    if any(dif > 0):
        dif = np.maximum(dif, [0, 0, 0])
        mod = dif % 2
        pad_1 = dif // 2
        pad_2 = pad_1 + mod
        if multimodal:
            pad_1 = [0] + pad_1.tolist()
            pad_2 = [0] + pad_2.tolist()
        pad = tuple(zip(pad_1, pad_2))
        im = np.pad(im, pad, 'reflect')
    return im, pad


def save_image(prediction, outpath, affine):
    new_pred = nib.Nifti1Image(prediction.numpy(), affine)
    new_pred.set_data_dtype(np.uint8)
    nib.save(new_pred, outpath)
