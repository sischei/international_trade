#!/bin/bash -l
#SBATCH --job-name=DEQ_1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=s995
#SBATCH --output=DEQ_1.%j.out
#SBATCH --error=DEQ_1.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load cray-python/3.6.5.7
module load daint-gpu
module load TensorFlow/2.2.0-CrayGNU-19.10-cuda-10.1.168
source venv/bin/activate
export PYTHONPATH=/scratch/snx3000/simonsch/DSGE_DEQ/src/sudden_stop/venv/lib/python3.6/site-packages:$PYTHONPATH

srun python run_deepnet.py 
