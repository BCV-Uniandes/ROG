import os
import shutil
import numpy as np
from joblib import Parallel, delayed

from utils import read_json, preprocess, cases_list


def Preprocess_datasets(out_dir, root, workers):
    tasks = [
        x for x in os.listdir(root)
        if os.path.isdir(os.path.join(root, x)) and x.startswith('Task')]
    tasks.sort()

    for x in tasks:
        out_task = os.path.join(out_dir, x)
        os.makedirs(out_task, exist_ok=True)
        os.makedirs(os.path.join(out_task, 'imagesTr'), exist_ok=True)
        os.makedirs(os.path.join(out_task, 'imagesTs'), exist_ok=True)
        os.makedirs(os.path.join(out_task, 'labelsTr'), exist_ok=True)

        dataset, limits, stats, spacing, CT = read_json(x, root)
        affine = np.diag(spacing + [1])
        spacing = np.asarray(spacing)
        args = {'task': os.path.join(root, x), 'spacing': spacing,
                'limits': limits, 'stats': stats, 'path': out_task,
                'affine': affine, 'CT': CT}

        shutil.copyfile(
            os.path.join(root, x, 'dataset.json'),
            os.path.join(out_task, 'dataset.json'))
        print('----- Processing training set -----')
        patientsTr = cases_list(dataset, out_task, 'imagesTr', 'training')
        Parallel(n_jobs=workers)(delayed(preprocess)(
            i['image'], args, lb=i['label']) for i in patientsTr)

        print('----- Processing test set -----')
        patientsTs = cases_list(dataset, out_task, 'imagesTs', 'test')
        Parallel(n_jobs=workers)(delayed(preprocess)(i, args) for i in patientsTs)
