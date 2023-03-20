#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:39:03 2023

@author: macbook
"""

import os, sys
sys.path.append("..")
sys.path.append('.')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src import distributions
import torch.nn.functional as F

from src.resnet2 import ResNet_D
from src.unet import UNet

from src.tools import unfreeze, freeze
from src.tools import weights_init_D
from src.tools import load_dataset, get_pushed_loader_stats
from src.tools import load_dataset, get_loader_stats
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_random_images, plot_images

from copy import deepcopy
import json

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from src.tools import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def stats_calcualtion(DATASET_LIST):
    DEVICE_ID = 0

    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_ID}')


    for DATASET, DATASET_PATH, IMG_SIZE in DATASET_LIST:
        print('Processing {}'.format(DATASET))
        sampler, test_sampler = load_dataset(DATASET, DATASET_PATH, img_size=IMG_SIZE)
        print('Dataset {} loaded'.format(DATASET))
    
        mu, sigma = get_loader_stats(test_sampler.loader)
        print('Trace of sigma: {}'.format(np.trace(sigma)))
        stats = {'mu' : mu.tolist(), 'sigma' : sigma.tolist()}
        print('Stats computed')
    
        filename = '{}_{}_test.json'.format(DATASET, IMG_SIZE)
        with open(filename, 'w') as fp:
            json.dump(stats, fp)
        print('States saved to {}'.format(filename))

if __name__ == '__main__':
    
    DATASET_LIST = [
         ('handbag', './handbag_64.hdf5', 64),
         ('shoes', './shoes_64.hdf5', 64)]
    
    stats_calcualtion(DATASET_LIST)
