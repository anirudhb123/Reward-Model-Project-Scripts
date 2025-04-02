#!/bin/bash
#SBATCH --job-name=my_script           # Job name
#SBATCH --output=output.log            # Output file
#SBATCH --error=error.log              # Error file
#SBATCH --time=12:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4              # Number of CPUs
#SBATCH --mem=16G                      # Memory per node

cd /mnt/nlpgridio3/data/anirudh2/
export OPENAI_API_KEY="PLACEHOLDER"
INPUT_PATH=data/GEMMA_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled_elaboration.jsonl
python3 main/elaboration_labeling.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
--model_name=gpt-4
