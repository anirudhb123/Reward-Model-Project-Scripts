#!/bin/bash
#SBATCH --mem=100G        
#SBATCH --nodelist=nlpgpu05
#SBATCH --gres=gpu:1 

# Activate your virtual environment.
source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate

# Set up cache directories.
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/

# Change to your working directory.
cd /mnt/nlpgridio3/data/anirudh2/

# Run the evaluation script

#python -u main/rewardbench_eval.py \
#  --model Skywork/Skywork-Reward-Gemma-2-27B-v0.2 \
#  --peft_adapter abharadwaj123/skywork-27b-fine-tuned \
#  --batch_size 8 \
#  --max_length 512 \
#  --torch_dtype bfloat16

python -u main/rewardbench_eval.py \
  --model Skywork/Skywork-Reward-Gemma-2-27B-v0.2 \
  --peft_adapter abharadwaj123/skywork-27b-fine-tuned \
  --batch_size 8 \
  --max_length 512 \
  --torch_dtype bfloat16