#!/bin/bash
#PBS -P CFP03-CF-088
#PBS -j oe
#PBS -k oed
#PBS -N verl-agent-A_Step_L_token_gigpo
#PBS -q auto
#PBS -l select=1:ncpus=64:ngpus=4:mem=500gb
#PBS -l walltime=18:00:00

cd $PBS_O_WORKDIR;

# 以下是在计算节点上执行的命令
echo "开始运行作业..."
nvidia-smi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent
#conda activate verl-agent-webshop
export WANDB_API_KEY=7ec99c214723e78aa44b8aba7f745a4797b12f53
# 运行你的程序
# bash /home/svu/vanzl/verl-agent/examples/advanced_grpo_trainer/run_alfworld_advanced_grpo.sh
# bash /home/svu/vanzl/verl-agent/examples/gspo_trainer/run_alfworld_adv_S_loss_S.sh
# bash /home/svu/vanzl/verl-agent/examples/gigpo_trainer/run_advanced_gigpo_alfworld.sh
# bash /home/svu/vanzl/verl-agent/examples/gspo_trainer/run_alfworld_adv_T_loss_S.sh
# bash /home/svu/vanzl/verl-agent/examples/gspo_trainer/run_alfworld_adv_Traj_loss_S.sh
# bash /home/svu/vanzl/verl-agent/examples/gigpo_trainer/run_webshop.sh



bash /home/svu/vanzl/verl-agent/examples/grpo_trainer/test.sh
num_runs=1
for ((i=0; i<num_runs; i++))
do
    # alfworld agent
    #bash /home/svu/vanzl/verl-agent/examples/gtpo_trainer/run_alfworld_adv_S_loss_Traj.sh
    #bash /home/svu/vanzl/verl-agent/examples/gtpo_trainer/run_alfworld_adv_T_loss_Traj.sh
    #bash /home/svu/vanzl/verl-agent/examples/gtpo_trainer/run_alfworld_adv_Traj_loss_Traj.sh

    # math
    #bash /home/svu/vanzl/verl-agent/examples/grpo_trainer/run_math_A_step_L_token.sh
    #bash /home/svu/vanzl/verl-agent/examples/grpo_trainer/run_math_A_token_L_token.sh
    #bash /home/svu/vanzl/verl-agent/examples/gspo_trainer/run_math_A_step_L_step.sh
    #bash /home/svu/vanzl/verl-agent/examples/gspo_trainer/run_math_A_token_L_step.sh
done
