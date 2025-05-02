#!/bin/bash

cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh
#export OPENAI_API_KEY=""
INPUT_PATH=data/all_data_baseline.jsonl
BIAS=hedging
OUTPUT_PATH=data/perturbations/all_data_perturbed_${BIAS}.jsonl
python3 main/generate_perturbed_responses_${BIAS}.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}