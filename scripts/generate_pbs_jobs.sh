#!/bin/bash
# =============================================================================
# 生成所有待跑实验的 PBS job 文件
#
# 使用方式:
#   1. bash scripts/generate_pbs_jobs.sh          # 生成所有 job 文件
#   2. bash scripts/submit_all.sh                 # 一键提交所有 job
#   3. bash scripts/submit_phase.sh 1             # 只提交某个 phase
#
# 远程集群路径: /home/svu/vanzl/verl-agent
# =============================================================================

set -euo pipefail

REPO_ROOT="/home/svu/vanzl/verl-agent"
JOB_DIR="$REPO_ROOT/jobs"
mkdir -p "$JOB_DIR"

# 清理旧 job 文件
rm -f "$JOB_DIR"/phase*.pbs

generate_pbs() {
    local phase="$1"
    local name="$2"
    local script="$3"
    local seed="$4"
    local conda_env="$5"       # verl-agent or verl-agent-webshop
    local walltime="$6"        # e.g. 24:00:00

    local job_name="${name}_s${seed}"
    local job_file="$JOB_DIR/phase${phase}_${job_name}.pbs"

    cat > "$job_file" << PBSEOF
#!/bin/bash
#PBS -P CFP03-CF-088
#PBS -j oe
#PBS -k oed
#PBS -N ${job_name}
#PBS -q auto
#PBS -l select=1:ncpus=64:ngpus=4:mem=500gb
#PBS -l walltime=${walltime}

cd \$PBS_O_WORKDIR;

echo "=========================================="
echo "Job: ${job_name}"
echo "Phase: ${phase}"
echo "Script: ${script}"
echo "Seed: ${seed}"
echo "Start: \$(date)"
echo "=========================================="

nvidia-smi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${conda_env}
export WANDB_API_KEY=7ec99c214723e78aa44b8aba7f745a4797b12f53

bash ${REPO_ROOT}/${script} env.seed=${seed}

echo "Finished: \$(date)"
PBSEOF

    echo "Generated: $job_file"
}

# =============================================================================
# Phase 1: ALFWorld L_step 重跑 (CLIP BUG fix, 最高优先级)
# 9 runs, ~150 epochs each
# =============================================================================
echo "=== Phase 1: ALFWorld L_step 重跑 (9 jobs) ==="
for seed in 0 1 2; do
    generate_pbs 1 "alf_Astep_Lstep"  "examples/gspo_trainer/run_alfworld_adv_S_loss_S.sh"    "$seed" "verl-agent" "24:00:00"
    generate_pbs 1 "alf_Atoken_Lstep" "examples/gspo_trainer/run_alfworld_adv_T_loss_S.sh"    "$seed" "verl-agent" "24:00:00"
    generate_pbs 1 "alf_Atraj_Lstep"  "examples/gspo_trainer/run_alfworld_adv_Traj_loss_S.sh" "$seed" "verl-agent" "24:00:00"
done

# =============================================================================
# Phase 2: ALFWorld L_traj 新跑 + 补 seed
# 8 runs
# =============================================================================
echo "=== Phase 2: ALFWorld L_traj + 补 seed (8 jobs) ==="

# A_step × L_token: 补 1 seed
generate_pbs 2 "alf_Astep_Ltoken"  "examples/grpo_trainer/run_alfworld_A_Step_L_token.sh"    2 "verl-agent" "24:00:00"

# A_step × L_traj: 3 seeds
for seed in 0 1 2; do
    generate_pbs 2 "alf_Astep_Ltraj"  "examples/gtpo_trainer/run_alfworld_adv_S_loss_Traj.sh"  "$seed" "verl-agent" "24:00:00"
done

# A_token × L_traj: 3 seeds
for seed in 0 1 2; do
    generate_pbs 2 "alf_Atoken_Ltraj" "examples/gtpo_trainer/run_alfworld_adv_T_loss_Traj.sh"  "$seed" "verl-agent" "24:00:00"
done

# A_traj × L_traj: 补 1 seed
generate_pbs 2 "alf_Atraj_Ltraj"  "examples/gtpo_trainer/run_alfworld_adv_Traj_loss_Traj.sh" 2 "verl-agent" "24:00:00"

# =============================================================================
# Phase 3: WebShop L_step 重跑 (CLIP BUG fix)
# 12 runs, ~250 epochs each
# =============================================================================
echo "=== Phase 3: WebShop L_step 重跑 (12 jobs) ==="
for seed in 0 1 2; do
    generate_pbs 3 "web_Astep_Lstep"  "examples/gspo_trainer/run_webshop_A_Step_L_Step.sh"     "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 3 "web_Atoken_Lstep" "examples/gspo_trainer/run_webshop_A_token_L_Step.sh"    "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 3 "web_Atraj_Lstep"  "examples/gspo_trainer/run_webshop_A_traj_L_Step.sh"     "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 3 "web_Agigpo_Lstep" "examples/gigpo_trainer/run_webshop_A_step_L_step.sh"    "$seed" "verl-agent-webshop" "48:00:00"
done

# =============================================================================
# Phase 4: WebShop L_traj 新跑
# 12 runs
# =============================================================================
echo "=== Phase 4: WebShop L_traj 新跑 (12 jobs) ==="
for seed in 0 1 2; do
    generate_pbs 4 "web_Astep_Ltraj"  "examples/gtpo_trainer/run_webshop_A_Step_L_Traj.sh"     "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 4 "web_Atoken_Ltraj" "examples/gtpo_trainer/run_webshop_A_token_L_Traj.sh"    "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 4 "web_Atraj_Ltraj"  "examples/gtpo_trainer/run_webshop_A_traj_L_Traj.sh"     "$seed" "verl-agent-webshop" "48:00:00"
    generate_pbs 4 "web_Agigpo_Ltraj" "examples/gtpo_trainer/run_webshop_A_gigpo_L_Traj.sh"    "$seed" "verl-agent-webshop" "48:00:00"
done

echo ""
echo "=== 总计: $(ls $JOB_DIR/phase*.pbs | wc -l) 个 PBS job 文件 ==="
echo "文件位置: $JOB_DIR/"
echo ""
echo "提交方式:"
echo "  全部提交:     bash scripts/submit_all.sh"
echo "  按 phase 提交: bash scripts/submit_phase.sh 1"
