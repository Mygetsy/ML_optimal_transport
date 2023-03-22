
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
import h5py

from src import distributions
import torch.nn.functional as F

from src.resnet2 import ResNet_D
from src.unet import UNet

from .tools import unfreeze, freeze
from .tools import weights_init_D
from .tools import load_dataset, get_pushed_loader_stats
from .tools import load_dataset, get_loader_stats
from .fid_score import calculate_frechet_distance
from .plotters import plot_random_images, plot_images

from copy import deepcopy
import json

def sampler_to_hdf5(X_sampler, T, save_dir, BATCH_SIZE=64, num_imgs=45000):
    '''
    X_sampler - sampler with loaded images
    T - trained nn
    save_dir - directory where hdf5 file would be saved
    BATCH_SIZE - batch size which was provided to T
    num_imgs - expected number of images  
    '''
    loader = X_sampler.loader
    n_iters = len(loader)
    n_image=0
    h5_file = h5py.File(save_dir, 'w')
        
    data = h5_file.create_dataset('imgs', shape=(num_imgs, 64, 64, 3), dtype=np.int64)
    freeze(T)
    for iter in range(n_iters-1):

        x = X_sampler.sample(BATCH_SIZE)
        T_x = T(x)
        imgs = (T_x.to('cpu').permute(0, 2, 3, 1).mul(0.5).add(0.5).numpy().clip(0, 1) * 255).astype(np.int64)
        for img in imgs:
            data[n_image] = img
            n_image+=1
            if (n_image) == num_imgs-1:
                break
    if n_image < num_imgs - 1:
        print(f'Warinig something wrong with batch size. Amount of pictures is less than {num_imgs}')
    h5_file.close()
