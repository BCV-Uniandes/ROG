import os
import glob
import json
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from scipy.ndimage import morphology, measurements

from utils import read_json


def parallel_sizes(image, labels):
    im = nib.load(image).get_fdata()
    struct = morphology.generate_binary_structure(3, 3)
    lb_sizes = []
    vol = []
    for lb in range(1, labels):
        # Find all connected components of category lb
        category = im == lb
        category, _ = measurements.label(category, struct)
        _, sizes = np.unique(category, return_counts=True)
        order = np.argsort(sizes)
        if len(sizes) > 1:
            # filter out the very small ones
            big = np.argwhere(sizes > sizes[order[-2]] * 0.05)
            vol_lb = []
            for obj in big[1:]:  # 0 is the background
                # find the bounding box location
                element = category == obj
                location = measurements.find_objects(element)
                coords = []
                for i in location[0]:
                    coords.append(i.stop - i.start)
                vol_lb.append(element.sum())
        lb_sizes.append(coords)
        vol.append(sum(vol_lb) / len(vol_lb))
    return im.shape, np.asarray(lb_sizes), vol


def Calcualte_sizes(root, num_workers):
    tasks = [
        x for x in os.listdir(root)
        if os.path.isdir(os.path.join(root, x)) and x.startswith('Task')]
    tasks.sort()

    for task in tasks:
        file_name = os.path.join('../..', 'Tasks', task, 'dataset_prueba.json')
        if os.path.isfile(file_name):
            print('Sizes of task {} already calculated'.format(task))
            continue

        print('Calculating sizes of task {}'.format(task))
        dataset, _, stats, _, _ = read_json(task, root)
        labels = len(dataset['labels'])

        files = glob.glob(os.path.join(root, task, 'labelsTr/*.nii.gz'))
        features = Parallel(n_jobs=num_workers)(delayed(parallel_sizes)(
            i, labels) for i in files)
        mean_size, sizes, volumes = [list(x) for x in zip(*features)]
        mean_size = np.median(mean_size, 0)
        mean_size[-1] = np.minimum(mean_size[-1], 160)
        sizes = np.round(np.mean(sizes, 0))
        volumes = np.round(np.mean(volumes, 0))
        order = np.argsort(volumes)

        results = {
            'mean_size': list(mean_size),
            'small_size': list(sizes[order[0]]),
            'volume_small': volumes[order[0]],
            'big_size': list(sizes[order[-1]]),
            'volume_big': volumes[order[-1]],
            'modality': dataset['modality'],
            'labels': dataset['labels']}

        with open(file_name, 'w') as outfile:
            json.dump(results, outfile, indent=4)
