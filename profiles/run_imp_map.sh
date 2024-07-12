#!/bin/bash -l                                                                                                       

#SBATCH --job-name="QD_I"
#SBATCH --time=30:00:00
#SBATCH --partition=compute-p2

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load miniconda3

srun python imp_map.py