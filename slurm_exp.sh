#!/usr/bin/env bash
##SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=15G
####SBATCH --exclusive
#SBATCH --gres=gpu:gtx1080:1
#####SBATCH --chdir=/home/$USER/qjobs
#SBATCH --output=cluster_cifar-%j.qout


module load cuda-9.0 python-36


python exp_lenk.py --n-processes 30
