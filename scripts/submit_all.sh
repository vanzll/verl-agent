#!/bin/bash
# 一键提交所有 PBS job
# 用法: bash scripts/submit_all.sh

set -euo pipefail

JOB_DIR="$(cd "$(dirname "$0")/.." && pwd)/jobs"

if [ ! -d "$JOB_DIR" ] || [ -z "$(ls $JOB_DIR/phase*.pbs 2>/dev/null)" ]; then
    echo "没有找到 job 文件，请先运行: bash scripts/generate_pbs_jobs.sh"
    exit 1
fi

echo "=== 提交所有 PBS jobs ==="
for f in $(ls "$JOB_DIR"/phase*.pbs | sort); do
    job_name=$(basename "$f" .pbs)
    echo -n "Submitting $job_name ... "
    qsub "$f"
done

echo ""
echo "=== 全部提交完成 ==="
echo "查看队列: qstat -u \$USER"
