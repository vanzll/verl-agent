#!/bin/bash
# Exercise 1 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=56:ngpus=2:mem=100G
#PBS -l walltime=12:00:00
#PBS -P 13014347 
#PBS -N verl-agent_naive_grpo
#PBS -o /home/users/astar/ares/yux5/scratch/verl-agent/log
# Commands start here
#cd ${PBS_O_WORKDIR}
cd /home/users/astar/ares/yux5/scratch/verl-agent
#cd $PBS_O_WORKDIR || exit $? #Change current directory to submission directory
nvidia-smi
#source /home/users/astar/ares/yux5/conda_initialize.sh
#which python
#pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
source /home/users/astar/ares/yux5/conda_initialize.sh
conda activate verl-agent
#python main.py
#bash ./examples/grpo_trainer/run_alfworld.sh
#bash ./examples/advanced_grpo_trainer/run_alfworld_advanced_grpo.sh
bash ./examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh
