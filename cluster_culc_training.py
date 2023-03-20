#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:38:41 2023

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

from tqdm import tqdm
from IPython.display import clear_output

import wandb # <--- online logging of the results
from src.tools import fig2data, fig2img # for wandb


# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)





def main():
    DEVICE_IDS = [0]

    DATASET1, DATASET1_PATH = 'shoes', './shoes_64.hdf5'
    DATASET2, DATASET2_PATH = 'handbag', './handbag_64.hdf5'
    
    # DATASET1, DATASET1_PATH = 'celeba_female', '../../data/img_align_celeba'
    # DATASET2, DATASET2_PATH = 'aligned_anime_faces', '../../data/aligned_anime_faces'
    
    T_ITERS = 10
    f_LR, T_LR = 1e-4, 1e-4
    IMG_SIZE = 64
    
    BATCH_SIZE = 128
    
    PLOT_INTERVAL = 100
    COST = 'mse' # Mean Squared Error
    CPKT_INTERVAL = 500
    MAX_STEPS = 100001
    SEED = 0x000000
    
    EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}'
    OUTPUT_PATH = '../checkpoints/{}/{}_{}_{}/'.format(COST, DATASET1, DATASET2, IMG_SIZE)

    config = dict(
    DATASET1=DATASET1,
    DATASET2=DATASET2, 
    T_ITERS=T_ITERS,
    f_LR=f_LR, T_LR=T_LR,
    BATCH_SIZE=BATCH_SIZE
    )
        
    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
    torch.manual_seed(SEED); np.random.seed(SEED)
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    filename = './stats/{}_{}_test.json'.format(DATASET2, IMG_SIZE)
    with open(filename, 'r') as fp:
        data_stats = json.load(fp)
        mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
    del data_stats

    X_sampler, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    Y_sampler, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
        
    torch.cuda.empty_cache(); gc.collect()
    # clear_output()


    f = ResNet_D(IMG_SIZE, nc=3).cuda()
    f.apply(weights_init_D)
    
    T = UNet(3, 3, base_factor=48).cuda()
    
    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)
        
    print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))


    torch.manual_seed(0xBADBEEF); np.random.seed(0xBADBEEF)
    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    X_test_fixed = X_test_sampler.sample(10)
    Y_test_fixed = Y_test_sampler.sample(10)
    
    fig, axes = plot_images(X_fixed, Y_fixed, T)
    fig, axes = plot_random_images(X_sampler, Y_sampler, T)
    fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
    fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
    plt.savefig('sart_fig_test.png')
    
    wandb.init(name=EXP_NAME, project='ml_ot_cluster', entity='mygetsy16',
               config=config)
    
    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)
    
    for step in range(MAX_STEPS):
    # T optimization
        unfreeze(T); freeze(f)
        for t_iter in range(T_ITERS): 
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE)
            T_X = T(X)
            if COST == 'mse':
                T_loss = F.mse_loss(X, T_X).mean() - f(T_X).mean()
            else:
                raise Exception('Unknown COST')
            T_loss.backward(); T_opt.step()
        del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()
    
        # f optimization
        freeze(T); unfreeze(f)
        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            T_X = T(X)
        Y = Y_sampler.sample(BATCH_SIZE)
        f_opt.zero_grad()
        f_loss = f(T_X).mean() - f(Y).mean()
        f_loss.backward(); f_opt.step();
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        print({f'f_loss' : f_loss.item()}, step) 
        del f_loss, Y, X, T_X; gc.collect(); torch.cuda.empty_cache()
            
        if step % PLOT_INTERVAL == 0:
            print('Plotting')
            clear_output(wait=True)
            
            fig, axes = plot_images(X_fixed, Y_fixed, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            # plt.show(fig); plt.close(fig) 
            
            fig, axes = plot_random_images(X_sampler,  Y_sampler, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            # plt.show(fig); plt.close(fig) 
            
            fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            # plt.show(fig); plt.close(fig) 
            
            fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            # plt.show(fig); plt.close(fig) 
        
        if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
            freeze(T); 
            
            print('Computing FID')
            mu, sigma = get_pushed_loader_stats(T, X_test_sampler.loader)
            fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
            wandb.log({f'FID (Test)' : fid}, step=step)
            print({f'FID (Test)' : fid}, step)
            del mu, sigma
            
            torch.save(T.state_dict(), 'T.pt')
            torch.save(f.state_dict(), os.path.join(OUTPUT_PATH, f'f_{SEED}_{step}.pt'))
            torch.save(f_opt.state_dict(), os.path.join(OUTPUT_PATH, f'f_opt_{SEED}_{step}.pt'))
            torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_{step}.pt'))
        
        gc.collect(); torch.cuda.empty_cache()

    


if __name__ == '__main__':
    main()


















