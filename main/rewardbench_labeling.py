r"""Determine which response is more structured in the easy chat subsets using RewardBench’s data.

Example usage:

OUTPUT_PATH=data/reward_bench_structure_labels.jsonl
python3 main/rewardbench_labeling.py \
  --output_path=${OUTPUT_PATH} \
  --model_name=gpt-4
"""

import json
import os
import sys
import re
from absl import app
from absl import flags
import tqdm

# Add parent path so that models and utils can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from datasets import load_dataset

# Import RewardBench functions to load evaluation data.
from rewardbench import load_eval_dataset, get_conv_template
from transformers import AutoTokenizer

# Define command-line flags.
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file with structure scoring results."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for structure scoring."
)

# Define the easy chat subsets.
# (Update these strings if your RewardBench examples use different subset names.)
EASY_CHAT_SUBSETS = {
    "alpacaeval-easy",
    "alpacaeval-length",
    "alpacaeval-hard",
    "mt-bench-easy",
    "mt-bench-medium"
}

def append_outputs(outputs, output_path):
    """Append the list of outputs to the file in JSONL format."""
    with open(output_path, "a") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

def sanitize_result(text):
    """
    Sanitize and determine the result based on the text output from the model.
    It searches for the keywords "chosen", "rejected", or "equal" (case-insensitive)
    and returns the corresponding sanitized result. If none are found, return an empty string.
    """
    lower_text = text.lower()
    if "chosen" in lower_text:
        return "chosen"
    elif "rejected" in lower_text:
        return "rejected"
    elif "equal" in lower_text:
        return "equal"
    else:
        return ""

def main(unused_argv) -> None:
    # === Load the RewardBench dataset ===
    #
    # For chat evaluation, we use a conversation template; here we use the default ("tulu").
    # You may adjust the tokenizer as needed (here we use a default pretrained model tokenizer, e.g., "gpt2").
    chat_template = "tulu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load the evaluation dataset from RewardBench. We keep only relevant columns.
    dataset, subset_metadata = load_eval_dataset(
        core_set=True,
        conv=get_conv_template(chat_template),
        tokenizer=tokenizer,
        keep_columns=["text_chosen", "text_rejected", "id"]
    )
    # Remove the id column since it is not needed.
    dataset = dataset.remove_columns("id")
    
    # RewardBench returns subset metadata separately. We add a new column "subset".
    # (Assume that subset_metadata is indexable and gives the subset label for each example.)
    # If this isn’t the case, adjust accordingly.
    dataset = dataset.add_column("subset", subset_metadata)
    
    # Filter to include only examples in the defined easy chat subsets.
    easy_chat_examples = [
        ex for ex in dataset if ex.get("subset") in EASY_CHAT_SUBSETS
    ]
    
    print(f"Found {len(easy_chat_examples)} examples in the easy chat subsets.\n")
    
    # === Instantiate the model based on the provided flag. ===
    model_name_lower = _MODEL_NAME.value.lower()
    if "gpt" in model_name_lower:
        model = models.GPT4(model_name=_MODEL_NAME.value)
    elif "gemini" in model_name_lower:
        model = models.Gemini(model_name=_MODEL_NAME.value)
    elif "claude" in model_name_lower:
        model = models.Claude(model_name=_MODEL_NAME.value)
    elif "jamba" in model_name_lower:
        model = models.Jamba(model_name=_MODEL_NAME.value)
    else:
        model = models.TogetherAI(model_name=_MODEL_NAME.value)
    
    # Create output file if it doesn't exist.
    if not os.path.exists(_OUTPUT_PATH.value):
        open(_OUTPUT_PATH.value, "w").close()
    
    # The prompt provided to the model.
    structure_classification_prompt = """
You are a text structure analysis expert. Your task is to decide which of the two responses is more structured, considering formatting, organization, and the presence of bullet points or lists.
If both responses appear equally structured, answer "equal".

Here is the input:

Prompt: [PROMPT]

Chosen Response: [CHOSEN]

Rejected Response: [REJECTED]

Provide your answer in the following format:

Result: [chosen/rejected/equal]
"""

    batch_outputs = []
    
    # Process each example with tqdm progress bar.
    for idx, ex in enumerate(tqdm.tqdm(easy_chat_examples, desc="Processing examples")):
        # For RewardBench data we have "text_chosen" and "text_rejected".
        # If you wish to provide a prompt, modify this accordingly.
        cur_prompt = (
            structure_classification_prompt
            .replace("[PROMPT]", "")  # No prompt available; replace with your desired text if needed.
            .replace("[CHOSEN]", ex.get("text_chosen", ""))
            .replace("[REJECTED]", ex.get("text_rejected", ""))
        )
        
        # Generate the model output.
        generated_output = model.generate(input_text=cur_prompt, max_len=100).strip()
        
        # Look for the line that contains "Result:" using a regular expression.
        result_line = ""
        for line in generated_output.split("\n"):
            if re.search(r"Result:", line, re.IGNORECASE):
                result_line = line
                break
        
        # Extract the result value after "Result:".
        if result_line:
            raw_result = result_line.split("Result:")[-1].strip()
        else:
            raw_result = ""
        
        # Sanitize the extracted result.
        result = sanitize_result(raw_result)
        
        record = {
            "prompt": "",  # no prompt provided; modify if needed
            "chosen": ex.get("text_chosen", ""),
            "rejected": ex.get("text_rejected", ""),
            "structure_result": result,
            "model_output": generated_output,
            "subset": ex.get("subset", "")
        }
        batch_outputs.append(record)
        
        # Write the batch to file every 50 records.
        if (idx + 1) % 50 == 0:
            append_outputs(batch_outputs, _OUTPUT_PATH.value)
            batch_outputs = []
    
    # Append any remaining records.
    if batch_outputs:
        append_outputs(batch_outputs, _OUTPUT_PATH.value)

if __name__ == "__main__":
    app.run(main)
