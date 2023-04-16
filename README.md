# Towards Robust General Medical Image Segmentation
This repository provides a PyTorch implementation of the Benchmark and baseline method presented in [Towards Robust General Medical Image Segmentation](https://arxiv.org/abs/2107.04263) ([MICCAI 2021](https://miccai2021.org/en/)). In this work, we propose a new benchmark to evaluate robustness in the context of the Medical Segmentation Decathlon (MSD) by extending the recent AutoAttack natural image classification framework to the domain of volumetric data segmentation, and we present a novel lattice architecture for RObust Generic medical image segmentation (ROG).

For mor information, please visit our [project website](https://bcv-uniandes.github.io/ROG_project/)

## Paper

[Towards Robust General Medical Image Segmentation](https://arxiv.org/abs/2107.04263),<br/>
[Laura Daza](https://lauradaza.github.io/Laura_Daza/)<sup>1</sup>, Juan C. Pérez<sup>1,2</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup>*<br/>
[MICCAI 2021](https://miccai2021.org/en/).<br><br>
<sup>1 </sup> Center for Research and Formation in Artificial Intelligence ([CINFONIA](https://cinfonia.uniandes.edu.co/)), Universidad de Los Andes. <br/>
<sup>2 </sup>King Abdullah University of Science and Technology (KAUST).<br/>

## Installation

### Cloning the repository

```bash
$ git clone git@github.com:BCV-Uniandes/ROG.git
$ cd ROG
$ python setup.py install
```

## Dataset Preparation

1. Download the Medical Segmentation Decathlon (MSD) Dataset from [here](http://medicaldecathlon.com/). Each task will be organized with the following structure:

```
TaskXX_TaskName
|_ imagesTr
|_ |_ *.nii.gz
|_ imagesTs
|_ |_ *.nii.gz
|_ labelsTr
|_ |_ *.nii.gz
|_ dataset.json
```

2. Set the `data_root`, `out_directory` and `num_workers` variables in the file [`data_preprocessing.py`](https://github.com/BCV-Uniandes/ROG/blob/main/data_preprocessing.py) and run the following command:

```
python data_preprocessing.py
```

If you want to use ROG for a different dataset you can store it in the same `data_root` folder, making sure that it follows the same name formating and organization as the MSD tasks.

## Training and evaluating the models

We train ROG on clean images and then fine-tune the models using Free AT:

```
# For the standard training
python main.py --task TASK_ID --gpu GPU_IDs --batch BATCH_SIZE --name OUTPUT_DIR

# For the Free AT fine tuning
python main.py --task TASK_ID --gpu GPU_IDs --batch BATCH_SIZE --name OUTPUT_DIR_FREE_AT --ft --pretrained OUTPUT_DIR --AT
```

Evaluating the models:

```
# Standard inference
python main.py --task TASK_ID --gpu GPU_IDs --batch BATCH_SIZE --name OUTPUT_DIR --test

# AutoAttack
python main.py --task TASK_ID --gpu GPU_IDs --batch BATCH_SIZE --name OUTPUT_DIR --test --adv
```

You can change the strength of the attacks by adjusting the magnitude (`--eps`) and number of iterations (`--adv_iters`).

## Citation

If you find our paper useful, please use the following BibTeX entry for citation:

```
@inproceedings{daza2021towards,
  title={Towards Robust General Medical Image Segmentation},
  author={Daza, Laura and P{\'e}rez, Juan C and Arbel{\'a}ez, Pablo},
  booktitle={MICCAI},
  year={2021}
}
```
