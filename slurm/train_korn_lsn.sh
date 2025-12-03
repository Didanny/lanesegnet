#!/bin/sh 
#SBATCH -n 9 
#SBATCH -N 1 
#SBATCH -p opengpu.p
#SBATCH -w korn 
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o slurm_logs/log_korn_lanesegnet.out
#SBATCH -e slurm_logs/err_korn_lanesegnet.out

export PYTHONPATH=$(pwd):$PYTHONPATH
./tools/dist_train.sh 4 --autoscale-lr --no-validate
