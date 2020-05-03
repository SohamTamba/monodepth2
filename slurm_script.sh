#!/bin/bash

#SBATCH --gres=gpu:p40:1
#SBATCH --time=28:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=Mono_Dist
#SBATCH --mail-type=END
##SBATCH --mail-user=sgt287@nyu.edu
#SBATCH --output=output.txt

module load anaconda3/5.3.1
source activate cv
conda install -n cv nb_conda_kernels

python train.py --save_frequency 9
