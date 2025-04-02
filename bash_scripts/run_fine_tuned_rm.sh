#!/bin/bash
#SBATCH --mem=128G        
#SBATCH --nodelist=nlpgpu05
#SBATCH --gres=gpu:1 

source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/

cd /mnt/nlpgridio3/data/anirudh2/

INPUT_PATH=data/all_data_perturbed_structure.jsonl
OUTPUT_PATH=data/structure_scored_fine_tuned_full.jsonl
MODEL_NAME=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
python3 -u main/run_fine_tuned_preference_model.py \
--model_name=${MODEL_NAME} \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}