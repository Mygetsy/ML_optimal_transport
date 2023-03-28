# Stacking for improving neural optimal transport based style transfer models

The repository contains the code implementation of the final project of the Skoltech Machine Learning 2023 course. The implementation is based on [Neural Optimal Transport](https://openreview.net/forum?id=d8CBRlWNkqH) work of Korotin A. et al. and their [PyTorch implementation](https://github.com/iamalexkorotin/NeuralOptimalTransport). 

## Using stacking to improve neural optimal transport based style transfer models.

### Problem statement 
The NOT approach can be applied to the style transfer problem as shown in the following figure. However, the mapped data contains some artifacts such as the purple shoe to bag transfer. 
<p align="center"><img src="pics/1itersb.PNG" width="550" /></p>

Therefore, we apply a stacking approach to improve the quality of the generated images. The quality of images estimated as FID metric.

### Datasets and Preprocessing

Two unpaired [datasets](https://github.com/junyanz/iGAN/blob/master/train_dcgan/README.md) were used, bags and shoes with 64x64 RGB images. Preprocessing of data sets for FID score is computed using ``culc_init_stats.py``. To create a new dataset from the trained NN and the existing dataset, ``mapping.py`` was used. 

### Example of results

The example images produced as a result of the stacking are presented in the following image. A more complete description of the results can be found in the ``report.pdf`` and ``presentation.pdf`` files.

<p align="center"><img src="pics/3stacking.PNG" width="550" /></p>

## Repository structure and code usage instructions
### Structure
The repository contains support functions in ```src/```. The ```notebooks/``` contains two notebooks explaining the NOT process (```NOT_training_strong.ipynb```) and the combination of the NOT and stacking processes (```NOT_training_stacking.ipynb```). The root folder also contains three Python scripts for preprocessing (```culc_init_stats.py```), using the trained NN to create a new dataset for the next iteration (```mapping.py```), and training the NN on a new dataset (```training.py```).

### Instructions
###### Python Scripts for Local Machines
For the first iteration of stacking, the running order should be ```culc_init_stats.py``` -> ```training.py```. For the next iteration of stacking, the scripts should be run in the following order: ```mapping.py``` -> ```training.py```. Modification of the input datasets is required in the above scripts. [Weights & Biases](https://wandb.ai), so login is required.

###### Notebooks for Collab
The self-explanatory notebook can be found at ```notebooks/NOT_training_stacking.ipynb``.

## Requirements
The code has been tested in the Premium Collab environment. The `torch== 1.9.0` version was used. The other libraries used were the latest update as of `24.03.2022`.

## Credits
- [NeuralOptimalTransport](https://github.com/iamalexkorotin/NeuralOptimalTransport) Github repo of the Neural Optimal Transport article by Korotin et al;
- [ADASE Group](https://github.com/adasegroup) for the great ML course;
- [Weights & Biases](https://wandb.ai) for machine learning developer tools;
