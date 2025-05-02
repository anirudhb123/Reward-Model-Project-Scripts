#!/bin/bash
#SBATCH --nodelist=nlpgpu[04-10]   
#SBATCH --gres=gpu:1     
#SBATCH --nodes=1              
#SBATCH --time=3-0

cd /mnt/nlpgridio3/data/anirudh2/
source set_keys.sh
#export OPENAI_API_KEY=""
BIAS=hedging
INPUT_PATH=data/GEMMA_training_sample.jsonl
OUTPUT_PATH=data/reward_model_training_labeled_data/GEMMA_sample_labeled_${BIAS}.jsonl
python3 -u main/${BIAS}_labeling.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=gpt-4
