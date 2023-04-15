import os
import csv
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage.morphology import binary_erosion
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")


def test(folder, root_dir, csv_file, classes):
    labels = pd.read_csv(csv_file)
    labels = labels['label'].tolist()
    images = os.listdir(folder)
    images.sort()
    labels.sort()

    patients = Parallel(n_jobs=10)(
        delayed(parallel_test)(images[j], labels[j], folder, root_dir, classes)
        for j in range(len(labels)))

    fields = ['Label'] + ['Dice_' + str(i) for i in range(1, classes)]
    fields = fields + ['Recall_' + str(i) for i in range(1, classes)]
    fields = fields + ['Precision_' + str(i) for i in range(1, classes)]
    mean = np.zeros([len(patients), (classes - 1) * 3])

    with open(folder + '.csv', 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fields)
        writer.writeheader()

        for idx, j in enumerate(patients):
            line = {field: datum for field, datum in zip(fields, j)}
            mean[idx] = np.asarray(j[1:])
            writer.writerow(line)

        mean = np.nanmean(mean, axis=0)
        last = ['mean'] + list(mean)
        writer.writerow({field: datum for field, datum in zip(fields, last)})
        outcsv.close()
    print(mean)


def read_image(path):
    im = nib.load(path)
    affine = im.affine
    im = im.get_data()
    return im, affine


def dice_score(im, lb):
    lb_f = np.ndarray.flatten(lb)
    im_f = np.ndarray.flatten(im)

    tps = np.sum(im * lb)
    fps = np.sum(im * (1 - lb))
    fns = np.sum((1 - im) * lb)
    labels = np.sum(lb_f)
    pred = np.sum(im_f)

    if labels == 0 and pred == 0:
        dice = 1
    else:
        dice = (2 * tps) / (2 * tps + fps + fns)
    rec = tps / (tps + fns)
    prec = tps / (tps + fps)
    return dice, rec, prec


def find_border(data):
    eroded = binary_erosion(data)
    border = np.logical_and(data, np.logical_not(eroded))
    return border


def get_coordinates(data, affine):
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    indices = np.vstack(np.nonzero(data))
    indices = np.vstack((indices, np.ones(indices.shape[1])))
    coordinates = np.dot(affine, indices)
    return coordinates[:3, :]


def eucl_max(nii1, nii2, affine):
    origdata1 = np.logical_not(np.logical_or(nii1 == 0, np.isnan(nii1)))
    origdata2 = np.logical_not(np.logical_or(nii2 == 0, np.isnan(nii2)))

    if origdata1.max() == 0 or origdata2.max() == 0:
        return np.NaN

    border1 = find_border(origdata1)
    border2 = find_border(origdata2)

    set1_coordinates = get_coordinates(border1, affine)
    set2_coordinates = get_coordinates(border2, affine)
    distances = cdist(set1_coordinates.T, set2_coordinates.T)
    mins = np.concatenate((np.amin(distances, axis=0),
                           np.amin(distances, axis=1)))
    return np.percentile(mins, 95)


def task(im, classes):
    tasks = []
    for i in range(classes):
        temp = im == i
        tasks.append(temp)
    return tasks


def parallel_test(im, lb, test_dir, root_dir, classes):
    dice = []
    precision = []
    recall = []
    # hausdorff = []

    name = im[:-4]
    im_path = os.path.join(test_dir, im)
    lb_path = os.path.join(root_dir, lb)

    im, affine = read_image(im_path)
    lb, _ = read_image(lb_path)
    lb = np.round(lb)

    im_task = task(im, classes)
    lb_task = task(lb, classes)

    for j in range(1, len(im_task)):
        dc, rec, prec = dice_score(im_task[j], lb_task[j])
        dice.append(dc)
        precision.append(prec)
        recall.append(rec)
        # hausdorff.append(eucl_max(im_task[j], lb_task[j], affine))
    return [name] + dice + recall + precision
