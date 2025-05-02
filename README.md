## Pipeline:


### Fine-Tuning

1. Generate perturbations for bias (```main/bash_scripts/run_perturbed.sh```)
2. Label training data subset (```main/bash_scripts/run_data_labeling.sh```)
3. Generate 750 counterfactuals from this subset as examples for fine tuning (```main/bash_scripts/run_counterfactual_generation.sh```)
4. In the proportion they appear in the labeled training data subset, sample+label examples from chatbot arena and use them as the additional 250 examples for fine tuning (```main/bash_scripts/run_chatblot_labeling.sh```)
5. Fine tune the model using 1000 examples (```main/bash_scripts/run_fine_tuning.sh```)


### Evaluation

1. Rewardbench evaluation (```main/bash_scripts/run_rewardbench_eval.sh```)
    a. Check output file for results
2. Score perturbed examples (```main/bash_scripts/run_fine_tuned_rm.sh```) 


## Bash Scripts

(Note: For these scripts, please define a `main/bash-scripts/set_keys.sh` that exports `OPENAI_API_KEY`, `HF_API_TOKEN`, and `WANDB_API_KEY`)

Brief overview of each script in this directory:

| Script                             |                   Description                                                            |
|------------------------------------|------------------------------------------------------------------------------------------|
| `main/bash_scripts/run_base.sh`                      | *(Not used currently)*                                                 |
| `main/bash_scripts/run_chatbot_labeling.sh`          | Selects examples from Chatbot Arena for fine-tuning (with counterfactual examples). |
| `main/bash_scripts/run_counterfactual_generation.sh` | Generates counterfactual examples to probe for bias.                   |
| `main/bash_scripts/run_data_labeling.sh`             | Labels training examples for the presence of bias.                     |
| `main/bash_scripts/run_fine_tuning.sh`               | Fine-tunes the reward model on generated counterfactuals.              |
| `main/bash_scripts/run_fine_tuned_rm.sh`             | Scores perturbed inputs using the fine-tuned reward model.             |
| `main/bash_scripts/run_perturbed.sh`                 | Generate perturbations for a particular bias (modify prompt in `generate_perturbed_responses.py` if using RATE).                        |
| `main/bash_scripts/run_rewardbench_eval.sh`          | Computes evaluation metrics on the RewardBench benchmark.              |
| `main/bash_scripts/run_rewardbench_labeling.sh`      | *(Not used currently; reserved for future RewardBench labeling)*       |
| `main/bash_scripts/run_rm.sh`                        | Runs the base (un-fine-tuned) reward model on a set of prompts.        |
