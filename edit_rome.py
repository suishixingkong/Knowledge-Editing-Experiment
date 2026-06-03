import sys
import os
import json
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
easyedit_root = r"./EasyEdit"
if easyedit_root not in sys.path:
    sys.path.append(easyedit_root)
from easyeditor import BaseEditor, ROMEHyperParams

model_path = "Qwen/Qwen3-1.7B"
data_path = "./test_set.json"
rome_path = "./rome/qwen3-1.7b.yaml"


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


def edit_by_rome(hparams, data, tokenizer):
    editor = BaseEditor.from_hparams(hparams)
    locality_input = {
        "neighborhood": {
            "prompt": [change_prompt(data["locality_prompt"])],
            "ground_truth": [data["locality_ground_truth"]],
        }
    }
    metrics, edited_model, _ = editor.edit(
        prompts=[change_prompt(data["prompt"])],
        rephrase_prompts=[change_prompt(data["rephrase_prompt"])],
        target_new=[data["target_new"]],
        subject=[data["subject"]],
        locality_inputs=locality_input,
        ground_truth=[data["ground_truth"]],
        sequential_edit=True,
    )
    response = []
    response.append(llm_request(data["prompt"], tokenizer, edited_model))
    response.append(llm_request(data["rephrase_prompt"], tokenizer, edited_model))
    response.append(llm_request(data["locality_prompt"], tokenizer, edited_model))
    return response


if __name__ == "__main__":
    with open(data_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, legacy=False
    )
    hparams = ROMEHyperParams.from_hparams(rome_path)

    i = 1
    for data in test_set:
        print(f"[{i}] 开始编辑 {data['prompt']} {data['ground_truth']} => {data['target_new']}")
        response = edit_by_rome(hparams, data, tokenizer)
        print(f"[{i}] 正在验证结果")
        print(f"[{i}] Prompt: {data['prompt']}, Response: {response[0]}")
        print(f"[{i}] Rephrase_Prompt: {data['rephrase_prompt']}, Response: {response[1]}")
        print(f"[{i}] Locality_Prompt: {data['locality_prompt']}, Response: {response[2]}")
        i += 1

