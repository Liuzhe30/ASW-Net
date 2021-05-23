#!/bin/bash

#SBATCH --job-name=unet
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 4
#SBATCH --mail-type=end
#SBATCH --mail-user=908166479@qq.com
#SBATCH --output=./unet.out
#SBATCH --error=./unet.err

python3 ./02-training-unet.py