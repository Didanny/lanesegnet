#!/bin/sh 
#SBATCH -n 9 
#SBATCH -N 1 
#SBATCH -p opengpu.p
#SBATCH -w poison 
#SBATCH --gres=gpu:4

export PYTHONPATH=$(pwd):$PYTHONPATH
./tools/dist_train.sh 4 --autoscale-lr
