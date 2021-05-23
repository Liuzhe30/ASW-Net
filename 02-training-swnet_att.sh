#!/bin/bash

#SBATCH --job-name=swnet_att
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 4
#SBATCH --mail-type=end
#SBATCH --mail-user=908166479@qq.com
#SBATCH --output=./swnet_att.out
#SBATCH --error=./swnet_att.err

python3 ./02-training-swnet_att.py