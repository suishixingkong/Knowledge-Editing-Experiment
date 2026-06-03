#!/bin/python

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json

model_name = "Qwen/Qwen3-1.7B"
data_path = "./test_set.json"


def change_prompt(prompt):
    inputs = """Q: The name of CEO of alibaba is?
A: Wu YongMing.
Q: The name of President of India is?
A: Smt.Droupadi Murmu.
"""
    inputs = inputs + f"Q: {prompt}?\nA: "
    return inputs


def llm_request(prompt, tokenizer, model):
    prompt = change_prompt(prompt)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        outputs[0][model_inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    for puc in ["\n", ",", ".", ";"]:
        if puc in text:
            text = text.split(puc)[0]
    text = text.strip()
    return text


with open(data_path, "r", encoding="utf-8") as f:
    test_set = json.load(f)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    "cuda"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, fix_mistral_regex=True, legacy=False
)

i = 1
for data in test_set:
    response = llm_request(data["prompt"], tokenizer, model)
    print(f"[{i}] Prompt: {data['prompt']}, Response: {response}")
    response = llm_request(data["locality_prompt"], tokenizer, model)
    print(f"[{i}] Locality_Prompt: {data['locality_prompt']}, Response: {response}")
    i += 1
