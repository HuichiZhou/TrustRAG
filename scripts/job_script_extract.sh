#!/bin/bash
#SBATCH --job-name=RAG_DEF
#SBATCH --output=logs/job_%j.txt    # Standard output
#SBATCH --error=logs/job_%j.txt      # Standard error
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH -c 8                      # number of cores (treats)
#SBATCH --gres=gpu:L40:1
#SBATCH --mail-user=tom.rahav@campus.technion.ac.il
#SBATCH --mail-type=NONE                # Send email on all events

# Activate conda environment
source .env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/tom.rahav/miniconda3/envs/trustrag

# Set parameters
DATASET="${DATASET:-hotpotqa}"           # ['nq','hotpotqa', 'msmarco']
EVAL_MODEL_CODE="${EVAL_MODEL_CODE:-contriever}" # ['contriever', 'contriever-ms', 'ance']
SCORE="${SCORE:-dot}"
METHOD="${METHOD:-drift}" # ['perplexity', 'drift']
TOPK="${TOPK:-5}"
huggingface-cli whoami
# Run the Python script with the parameters passed from the environment
python3 -u /home/tom.rahav/TrustRAG/get_thresh.py \
        --eval_dataset "$DATASET" \
        --score_function "$SCORE" \
        --eval_model_code "$EVAL_MODEL_CODE" \
        --method "$METHOD" \
        --top_k $TOPK

