#!/bin/bash
#
# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=M1_rel_tol_sweep # name of the job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00 # maximum duration of the job
#SBATCH --output=log.out # name of the output log file
#SBATCH --error=log.err # name of the error log file

# go to the folder containing the script
cd /mnt/shared/home/b31xo3/odl1_project/implementation/M1

# activate python enviroment
export PATH=~/miniconda3/bin:$PATH
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate b31xo_cuda

# launch the python script you wanted to create
python M1_rel_tol.py