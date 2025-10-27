#!/bin/bash
# Exercise 1 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=64:ngpus=4:mem=512G
#PBS -l walltime=12:00:00
#PBS -P 13014347 
#PBS -N verl-agent
#PBS -o log/
# Commands start here
#cd ${PBS_O_WORKDIR}
cd /home/users/astar/ares/yux5/scratch/verl-agent
#cd $PBS_O_WORKDIR || exit $? #Change current directory to submission directory
mkdir log
nvidia-smi

#source /home/users/astar/ares/yux5/conda_initialize.sh
#which python
#pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
conda activate verl-agent
#python main.py
bash ./examples/grpo_trainer/run_alfworld.sh

