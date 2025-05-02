# file: score_with_multiple_llms.py
import argparse
import random
import re
import pandas as pd

# ——— Providers’ SDKs ———
import google.generativeai as genai
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# configure your API keys (or set via env vars)
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
openai.api_key = "YOUR_OPENAI_API_KEY"
claude = Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")

def extract_judgement(text):
    m = re.search(r'\{"judgement": "([^"]+)"\}', text)
    return m.group(1) if m else "No judgement found"

def construct_prompt(query, r1, r2):
    return f"""
    You will be given a query issued by a real user to a language model. You will also be given two model responses to this query, and you will need to judge which response is better.

    IMPORTANT: You should produce the final judgement as a dictionary in precisely this format (with **): "**output: {{"judgement": "_"}}**", where you should fill in the spaces with either "Response 1" if Response 1 is better, "Response 2" if Response 2 is better or "Tie" if both responses are equally good or equally bad. Only the three choices "Response 1", "Response 2" and "Tie" are valid. Make note of the ** required to enclose the output dictionary. After generating the output, provide a brief justification of your judgement.

    Query: {query}

    Response 1: {r1}

    Response 2: {r2}

    Judgement:
"""

def call_gemini(prompt):
    resp = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)
    return resp.text

def call_gpt4o(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

def call_claude(prompt):
    resp = claude.completions.create(
        model="claude-3.7",
        prompt=HUMAN_PROMPT + prompt + AI_PROMPT,
        max_tokens_to_sample=512
    )
    return resp.completion

BACKENDS = {
    "gemini-2.5-pro": call_gemini,
    "gpt-4o":         call_gpt4o,
    "claude-3.7":     call_claude,
}

def generate_labels(model_name, infile, question_col, r1_col, r2_col, outfile):
    df = pd.read_csv(infile)
    results = []

    for _, row in df.iterrows():
        q = row[question_col]
        a1, a2 = row[r1_col], row[r2_col]

        # shuffle
        lab_and_resp = [("Response 1", a1), ("Response 2", a2)]
        random.shuffle(lab_and_resp)
        labels, resps = zip(*lab_and_resp)

        prompt = construct_prompt(q, resps[0], resps[1])
        raw = BACKENDS[model_name](prompt)
        j = extract_judgement(raw)

        # map back to “Response 1/2”
        chosen = None
        if j=="Tie":
            chosen="Tie"
        else:
            # whichever side the judgement picked
            chosen = labels[0] if j=="Response 1" else labels[1]

        results.append({
            question_col: q,
            r1_col: a1,
            r2_col: a2,
            "chosen": chosen,
            "raw_judgement": raw
        })

    out = pd.DataFrame(results)
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=BACKENDS.keys(), required=True)
    p.add_argument("--infile",    required=True)
    p.add_argument("--question",  required=True)
    p.add_argument("--r1",        required=True)
    p.add_argument("--r2",        required=True)
    p.add_argument("--outfile",   required=True)
    args = p.parse_args()

    generate_labels(
        model_name=args.model,
        infile=args.infile,
        question_col=args.question,
        r1_col=args.r1,
        r2_col=args.r2,
        outfile=args.outfile
    )
