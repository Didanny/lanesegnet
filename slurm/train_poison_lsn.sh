#!/bin/sh 
#SBATCH -n 9 
#SBATCH -N 1 
#SBATCH -p opengpu.p
#SBATCH -w poison 
#SBATCH --gres=gpu:4
#SBATCH -o slurm_logs/log_poison_lanesegnet.out
#SBATCH -e slurm_logs/err_poison_lanesegnet.out

export PYTHONPATH=$(pwd):$PYTHONPATH
./tools/dist_train.sh 4 --autoscale-lr --no-validate
