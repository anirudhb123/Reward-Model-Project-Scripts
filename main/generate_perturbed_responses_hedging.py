#!/usr/bin/env python3
"""
Generate specific vs. vague rewrites *on top* of your existing answers.

1) Uses ex["response"] as the direct baseline.
2) Rewrites that text into:
     • a clear, confident “specific” answer
     • a hedged, low-info “vague” answer
3) If ex["response"] is empty, falls back to your few-shot prompts.
"""
from __future__ import annotations
from absl import app, flags
import os, re, sys, json, tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import models                  # your repo’s wrapper
from data_utils import jsonl_utils

# ----------------------------- FLAGS --------------------------------- #
_INPUT_PATH  = flags.DEFINE_string("input_path",  "",  "Baseline file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "",  "Destination file.")
_MODEL_NAME  = flags.DEFINE_string("model_name",  "gpt-4o", "LLM backend.")

_MIN_CHARS   = flags.DEFINE_integer("min_chars",     5, "Min chars accepted.")
_MAX_RETRIES = flags.DEFINE_integer("max_retries",   3, "Retries per prompt.")
_BATCH_SIZE  = flags.DEFINE_integer("batch_size",   50, "Flush interval.")

# The key in each JSONL row that holds your original answer:
ORIG_KEY = "original_response"

# --------------------------- UPDATED FEW-SHOT PAIRS (V4 Style) -------------------------- #
FEW_SHOTS = [
    (
        "Is it unethical to work on building weapons?",
        # VAGUE (updated - more nuanced hedging)
        "Determining the ethics surrounding work on weapons development involves navigating a complex web of moral considerations and societal viewpoints. It's a deeply nuanced issue where individuals often hold differing perspectives based on their personal values, ethical frameworks, and understanding of concepts like defense and responsibility. Many factors contribute to how people approach this question, and it touches upon long-standing debates about the nature of conflict and the application of ethical principles in sensitive domains. There isn't a universally agreed-upon answer.",
        # SPECIFIC (unchanged)
        "The question of whether it is unethical to work on building weapons is subject to significant ethical debate, often framed by differing moral philosophies. One perspective, often rooted in deontology, argues that creating instruments designed to inflict harm is inherently wrong, regardless of the intended use. Conversely, a consequentialist or utilitarian viewpoint might assess the ethics based on the outcomes, arguing that building weapons can be ethical if it serves as a deterrent, prevents greater harm, or protects national security. Discussions also frequently involve the concept of individual responsibility within a larger system and the intent behind the development and potential deployment of such technologies."
    ),
    (
        "What is the idea behind Rotary Position Embeddings in transformers?",
        # VAGUE (updated - hints at technical nature without details)
        "Rotary Position Embeddings in transformers are related to the technical challenge of enabling these models to understand sequence order. Since transformers process data in a non-sequential way, various methods exist to incorporate positional context. RoPE represents a specific technique within this area, aiming to integrate information about where elements appear relative to each other. Understanding its role involves considering how transformer architectures handle input structure and the different strategies employed to encode positional awareness for effective processing.",
        # SPECIFIC (unchanged)
        "Rotary Position Embeddings (RoPE) in transformers are designed to inject positional information into the self-attention mechanism in a way that naturally captures relative positions between tokens. Unlike absolute position embeddings which add a fixed value based on a token’s position, RoPE works by rotating the query and key vectors based on their position. This rotational mechanism has the benefit of allowing the dot product, which is central to attention, to implicitly encode the distance or relative position between tokens, which can improve the model’s ability to handle sequences of varying lengths and extrapolate to longer sequences."
    ),
]

V_LABEL = "Empty Statement:"
S_LABEL = "Specific Answer:"

def _block(q: str, vague: str, specific: str) -> str:
    # This helper function remains the same
    return (
        f"Query: {q}\n"
        f"{V_LABEL} {vague}\n"
        f"{S_LABEL} {specific}\n\n"
    )

def vague_prompt(query: str) -> str:
    # Updated header with refined instructions (based on V4 discussion)
    header = (
        "Each pair below shows a clear, detailed **Specific Answer** and an intentionally "
        "vague **Empty Statement**. Your task is to generate a new Empty Statement for the given Query.\n\n"
        "The **Empty Statement** MUST:\n"
        " • Clearly reference the core subject matter of the Query.\n"
        " • Acknowledge the complexity, nuance, or multi-faceted nature of the topic (if applicable).\n"
        " • Allude generally to the *types* of factors, considerations, or perspectives involved (e.g., 'various factors', 'different viewpoints', 'technical considerations', 'personal values') WITHOUT detailing them.\n"
        " • Maintain a cautious and neutral tone.\n"
        " • Contain **NO** specific numbers, dates, detailed definitions, proper nouns (unless essential for topic identity like 'transformers'), step-by-step instructions, or concrete examples.\n"
        " • **AVOID** directly answering the question or providing any specific framework, solution, or conclusion.\n\n"
        f"For the NEW query, write ONLY the text that should follow '{V_LABEL}' (do not include the '{V_LABEL}' label itself).\n\n"
        "--- EXAMPLES ---\n"
    )
    # Use the updated FEW_SHOTS_V4 list
    shots = "".join(_block(q, v, s) for q, v, s in FEW_SHOTS_V4)
    # Assemble the final prompt for the new query
    return f"{header}{shots}--- NEW TASK ---\nQuery: {query}\n{V_LABEL}" # Added separator for clarity

# The specific_prompt function likely doesn't need changes for this task,
# assuming it's just used to generate the 'specific' counterpart using the same examples.
# If needed, ensure it uses the same FEW_SHOTS_V4 list for consistency.
def specific_prompt(query: str) -> str:
    header = (
        "Below are comparison pairs showing a vague 'Empty Statement' and a detailed 'Specific Answer'.\n"
        f"For the NEW query, write **only** the detailed text that would follow '{S_LABEL}' — no label, no markup.\n\n"
         "--- EXAMPLES ---\n"
    )
    # Use the same FEW_SHOTS_V4 list for consistency in the examples shown
    shots = "".join(_block(q, v, s) for q, v, s in FEW_SHOTS_V4)
    return f"{header}{shots}--- NEW TASK ---\nQuery: {query}\n{S_LABEL}"


def load_model(name: str):
    if "gpt"    in name: return models.GPT4(model_name=name)
    if "gemini" in name: return models.Gemini(model_name=name)
    if "claude" in name: return models.Claude(model_name=name)
    if "jamba"  in name: return models.Jamba(model_name=name)
    return models.TogetherAI(model_name=name)

def _clean(text: str) -> str:
    return text.replace(V_LABEL, "").replace(S_LABEL, "").strip()

def generate_nonblank(model, prompt: str, min_chars: int = None) -> str:
    target = min_chars or _MIN_CHARS.value
    last = ""
    for _ in range(_MAX_RETRIES.value):
        out = _clean(model.generate(prompt, max_len=512))
        last = out
        if len(out) >= target:
            return out
        prompt += "\n\nThat was too short. Please try again with a longer answer."
    return last or "Response unavailable."

def rewrite_prompt(query: str, original: str, style: str) -> str:
    """
    style in {"specific", "vague"}.
    """
    assert style in ("specific", "vague")
    role = (
        "a **Direct Answer** that is clear, confident, no hedging"
        if style == "specific"
        else
        "an **Empty / Vague Statement** that keeps the same information but "
        "adds hedging, meta-phrases, and avoids sounding definitive"
    )
    return f"""You are rewriting an answer while preserving all content.

QUESTION
{query}

ORIGINAL ANSWER
\"\"\"{original}\"\"\"

TASK
Rewrite the ORIGINAL ANSWER into {role}.
• Do **not** add or remove facts.
• Keep roughly the same length (±10 % tokens).
• Do not mention that you are rewriting.
Return only the rewritten text.
"""

def main(_):
    examples = jsonl_utils.read(_INPUT_PATH.value)
    model    = load_model(_MODEL_NAME.value)
    outputs  = []

    for idx, ex in enumerate(tqdm.tqdm(examples[:50], desc="Generating")):
        q    = ex["query"]
        orig = ex.get(ORIG_KEY, "").strip()

        if orig:
            # keep the original as the “direct” answer
            ex["base_response"] = orig

            # rewrite into a hedged / vague answer
            ex["perturbed_response"] = generate_nonblank(
                model,
                rewrite_prompt(q, orig, style="vague")
            )
        else:
            # fallback: generate both from scratch
            ex["base_response"]      = generate_nonblank(model, specific_prompt(q))
            ex["perturbed_response"] = generate_nonblank(model, vague_prompt(q))

        outputs.append(ex)

        if (idx + 1) % _BATCH_SIZE.value == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

    jsonl_utils.write(_OUTPUT_PATH.value, outputs)

if __name__ == "__main__":
    app.run(main)
