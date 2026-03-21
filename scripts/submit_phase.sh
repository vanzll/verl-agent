#!/bin/bash
# 按 phase 提交 PBS jobs
# 用法: bash scripts/submit_phase.sh <phase_number>
#   phase 1 = ALFWorld L_step 重跑
#   phase 2 = ALFWorld L_traj + 补 seed
#   phase 3 = WebShop L_step 重跑
#   phase 4 = WebShop L_traj 新跑

set -euo pipefail

PHASE=${1:?"Usage: $0 <1|2|3|4>"}
JOB_DIR="$(cd "$(dirname "$0")/.." && pwd)/jobs"

FILES=$(ls "$JOB_DIR"/phase${PHASE}_*.pbs 2>/dev/null)
if [ -z "$FILES" ]; then
    echo "Phase $PHASE: 没有找到 job 文件"
    exit 1
fi

COUNT=$(echo "$FILES" | wc -l)
echo "=== 提交 Phase $PHASE ($COUNT jobs) ==="

for f in $FILES; do
    job_name=$(basename "$f" .pbs)
    echo -n "Submitting $job_name ... "
    qsub "$f"
done

echo ""
echo "=== Phase $PHASE 提交完成 ==="
echo "查看队列: qstat -u \$USER"
