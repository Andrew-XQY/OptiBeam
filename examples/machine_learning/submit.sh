#!/bin/bash
#SBATCH --job-name=image_reconstruction_model_training  # Job name
#SBATCH --output=results/slurm/nn_training-%j.out                     # Output file
#SBATCH --error=results/slurm/nn_training-%j.err                      # Error file
#SBATCH --partition=gpu                                 # Partition name, adjust to your system
#SBATCH --nodes=1                                       # Number of nodes
#SBATCH --ntasks-per-node=1                             # Number of tasks per node
#SBATCH --gres=gpu:1                                    # Number of GPUs per task
#SBATCH --mem=90G                                       # Memory per node
#SBATCH --cpus-per-task=8                               # Number of CPU cores per task
#SBATCH --time=24:00:00                                 # Time limit hrs:min:sec

# Load necessary modules
module load apps/apptainer

# Run the container with the neural network training script
srun apptainer exec --nv --writable-tmpfs -B ./dataset:/app/dataset -B ./results:/app/results -B ./code:/app/code \
    ./tf_gpu_latest.sif /bin/bash -c "cd /app/code/examples/machine_learning && python training.py"
