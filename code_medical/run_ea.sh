#!/bin/bash -l
#SBATCH --job-name=ea_medical
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

unset SLURM_EXPORT_ENV

# ── Paths ──
PROJECT_DIR=/home/woody/iwi5/iwi5184h/Project_Medical
CODE_DIR=${PROJECT_DIR}/code_medical
DATA_DIR=${PROJECT_DIR}/data
CKPT_DIR=${CODE_DIR}/checkpoints

# ── Environment ──
module load python
module load cuda/12.9.0
source ~/.bashrc
conda activate medical_ea

# ── Create log directory ──
mkdir -p ${CODE_DIR}/logs

# ── Copy data to fast local SSD ──
echo "[$(date '+%H:%M:%S')] Copying data to \$TMPDIR ..."
cp -r ${DATA_DIR} ${TMPDIR}/data
echo "[$(date '+%H:%M:%S')] Data copy done. $(du -sh ${TMPDIR}/data | cut -f1)"

# ── Run EA ──
cd ${CODE_DIR}
echo "[$(date '+%H:%M:%S')] Starting EA evolution on ${SLURM_JOB_NUM_NODES} node(s), 8 GPUs"
echo "[$(date '+%H:%M:%S')] SLURM Job ID: ${SLURM_JOB_ID}"
echo "[$(date '+%H:%M:%S')] Checkpoint dir: ${CKPT_DIR}"

python main.py \
    --data_dir ${TMPDIR}/data \
    --N 50 \
    --IPC 50 \
    --G 30 \
    --num_gpus 8 \
    --checkpoint_dir ${CKPT_DIR} \
    --resume \
    --baselines baselines.json \
    --seed 2025

EXIT_CODE=$?
echo "[$(date '+%H:%M:%S')] Python exited with code ${EXIT_CODE}"

# If we were signaled (SIGUSR1), the job will be requeued by SLURM automatically
# The --resume flag ensures we pick up from the latest checkpoint
