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
from src.mapper import sampler_to_hdf5

from copy import deepcopy
import json

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from src.tools import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def mapping(INPUT_DATASET_TUPLE, T_ADRESS, FILE_PATH):
    DEVICE_ID = 0
    
    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_ID}')

    DATASET, DATASET_PATH, IMG_SIZE = INPUT_DATASET_TUPLE
    print('Processing {}'.format(DATASET))
    sampler, _ = load_dataset(DATASET, DATASET_PATH, img_size=IMG_SIZE)
    print('Dataset {} loaded'.format(DATASET))
    
    T = UNet(3, 3, base_factor=48).cuda()
    T.load_state_dict(torch.load(T_ADRESS))

    sampler_to_hdf5(sampler, T, FILE_PATH, num_imgs=50000)

if __name__ == '__main__':
    
    INPUT_DATASET_TUPLE = ('handbag', './handbag_64.hdf5', 64)
    T_ADRESS = '/home/d.panov/ML/checkpoints/mse/shoes_handbag_64/T_0_10000.pt'
    FILE_PATH = './shoes_map_iter_1.hdf5'
    
    mapping(INPUT_DATASET_TUPLE, T_ADRESS, FILE_PATH)
