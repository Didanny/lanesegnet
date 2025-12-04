#!/bin/sh 
#SBATCH -n 1
#SBATCH -N 1 
#SBATCH -p openlab.p
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00

git checkout dev-alternative-refinement
git stash apply
