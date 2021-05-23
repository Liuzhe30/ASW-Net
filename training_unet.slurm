#!/bin/bash

#SBATCH --job-name=unet
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 4
#SBATCH --mail-type=end
#SBATCH --mail-user=908166479@qq.com
#SBATCH --output=./data_IMC_FINAL/training_unet.out
#SBATCH --error=./data_IMC_FINAL/training_unet.err

python3 02-training.py