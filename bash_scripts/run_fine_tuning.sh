#!/bin/bash
#SBATCH --job-name=skywork_finetune
#SBATCH --nodelist=nlpgpu09
#SBATCH --gres=gpu:1              
#SBATCH --mem=256G               
#SBATCH --cpus-per-task=8        
#SBATCH --time=3-0

# Activate your scratch venv
source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

# Confirm you’re using the venv’s Python
echo "Using Python: $(which python)"

# Set up Hugging Face cache directories
export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/
export HF_TOKEN="PLACEHOLDER"

# Navigate to project directory
cd /mnt/nlpgridio3/data/anirudh2/

# Define training parameters
INPUT_PATH="data/GEMMA_counterfactuals.jsonl"
MODEL_REPO_ID="abharadwaj123/skywork-27b-fine-tuned-full"
BASE_MODEL_NAME="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=2e-5
USE_LORA=true
LORA_R=4
LORA_ALPHA=8
VALIDATION_SPLIT=0.1

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run fine‑tuning with the venv’s Python
python -u main/counterfactual_fine_tuning.py \
  --input_path=${INPUT_PATH} \
  --model_repo_id=${MODEL_REPO_ID} \
  --base_model_name=${BASE_MODEL_NAME} \
  --epochs=${EPOCHS} \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=${LEARNING_RATE} \
  --use_lora=${USE_LORA} \
  --lora_r=${LORA_R} \
  --lora_alpha=${LORA_ALPHA} \
  --validation_split=${VALIDATION_SPLIT}

echo "Fine‑tuning completed. Model pushed to Hugging Face Hub: ${MODEL_REPO_ID}"
