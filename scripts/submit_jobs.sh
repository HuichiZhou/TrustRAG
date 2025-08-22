#!/bin/bash

# Define arrays of parameters 
datasets=('nq' 'hotpotqa' 'mirage') # 'nq' 'hotpotqa' 'msmarco' 'mirage'
models_names=("meta-llama/Llama-3.1-8B-Instruct") # "mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct" "gpt-4o" "meta-llama/Llama-2-7b-chat-hf"
eval_models_names=("contriever" "contriever-ms" "ance") # "contriever" "contriever-ms" "ance" "minilm" "mpnet" "roberta"
attacks=("hotflip" "aggd") # "none" "LM_targeted" "pia" "hotflip" "aggd" "blackbox_hotflip"
removals=('drift') # 'none' 'drift' 'kmeans' 'kmeans_ngram' 'all' 'drift_kmeans_ngram' 'perplexity'
defenses=('none') # 'none' 'conflict' 'astute' 'instruct'
scores=('dot') # 'dot' 'cos_sim'
positions=('start' 'end') # 'start' 'end'
advs_per_query=(5)
num_tokens_to_remove=(30) # 0 5 10 15 20 25 30 35 40 45 50 55 56
lambdas=(5)

index=1  # Initialize counter

# Iterate over each combination of parameters
for dataset in "${datasets[@]}"; do
  for model_name in "${models_names[@]}"; do
    for eval_model_name in "${eval_models_names[@]}"; do
      for attack in "${attacks[@]}"; do
        for removal in "${removals[@]}"; do
          for defense in "${defenses[@]}"; do
            for score in "${scores[@]}"; do
              for position in "${positions[@]}"; do
                for apq in "${advs_per_query[@]}"; do
                  for ntr in "${num_tokens_to_remove[@]}"; do
                    for lambda in "${lambdas[@]}"; do
                      echo $index
                      # Submit the job with the current set of parameters
                      sbatch --export=DATASET="$dataset",MODEL_NAME="$model_name",EVAL_MODEL_CODE="$eval_model_name",ATTACK="$attack",REMOVAL="$removal",DEFENSE="$defense",SCORE="$score",ADV_A_POSITION="$position",ADV_PER_QUERY="$apq",NUM_TOKENS_TO_REMOVE="$ntr",LAMBDA="$lambda" scripts/job_script.sh
                      ((index++))  # Increment counter
                      sleep 1  # Brief pause to be kind to the scheduler
                      done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done