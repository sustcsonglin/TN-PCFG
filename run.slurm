#!/bin/bash
#SBATCH -J lcfrs_t
#SBATCH -p critical_a40
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 10:30:30
#SBATCH --gres=gpu:1
#SBATCH --output=pcfg2.out
#SBATCH --error=pcfg2.err


CUDA_HOME='/public/software/CUDA/cuda-11.4' python train.py --conf=config/arpcfg.yaml
