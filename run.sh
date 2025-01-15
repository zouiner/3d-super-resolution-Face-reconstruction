#!/bin/bash
#SBATCH --job-name=sr3d_16_32_model3             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=moonlight.dum1@gmail.com          # Where to send mail
#SBATCH --ntasks=1                             # Run a single task...
#SBATCH --cpus-per-task=1                      # ...with a single CPU
#SBATCH --mem=16gb                             # Job memory request
#SBATCH --time=3-00:00:00                      # Time limit (DD-HH:MM:SS)
#SBATCH --output=cuda_log/cuda_job_sr3d_16_32_model3_%j.log      # Standard output and error log
#SBATCH --partition=gpu                        # Select the GPU nodes... (, interactive, gpu , gpuplus)  
#SBATCH --gres=gpu:1                          # ...and the Number of GPUs
#SBATCH --account=its-gpu-2023                 # Run job under project <project>

module purge
module load GCC/12.2.0
module load Miniconda3 # for conda, if using venv you wont need this

# This is just printing stuff you don't really need these lines
echo `date`: executing gpu_test on host $HOSTNAME with $SLURM_CPUS_ON_NODE cpu cores echo 
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo I can see GPU devices $CUDA_VISIBLE_DEVICES
echo

source ~/.bashrc

conda activate 3dsr # Your conda environment here 
# or source your .venv environment here if using venv

# now run the python script
# python main_temp.py -p val -c config/sr_sr3_VGGF2_8_16_model2.yml -s 15
python main_temp.py -p train -c config/sr_sr3_VGGF2_16_32_model3.yml

# to run the command: sbatch run.sh