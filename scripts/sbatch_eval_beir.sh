#!/bin/bash
#SBATCH --job-name=RAG_DEF
#SBATCH --output=logs/job_%j.txt    # Standard output
#SBATCH --error=logs/job_%j.txt      # Standard error
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH -c 8                      # number of cores (treats)
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-user=tom.rahav@campus.technion.ac.il
#SBATCH --mail-type=NONE                # Send email on all events

# Activate conda environment
source .env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/tom.rahav/miniconda3/envs/trustrag

MODEL=ance
SCORE=dot
DATASET=msmarco
# Execute your Python script
python evaluate_beir.py --model_code $MODEL \
                        --score_function $SCORE \
                        --top_k 20 \
                        --dataset $DATASET \
                        --per_gpu_batch_size 16 \
                        --result_output "results/beir_results/$DATASET-$MODEL-$SCORE.json"


