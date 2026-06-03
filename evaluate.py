import sys
import os
import json
from transformers import AutoTokenizer
from edit_rome import edit_by_rome

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
easyedit_root = r"./EasyEdit"
if easyedit_root not in sys.path:
    sys.path.append(easyedit_root)
from easyeditor import BaseEditor, ROMEHyperParams

model_path = "Qwen/Qwen3-1.7B"
data_path = "./test_set.json"
rome_path = "./rome/qwen3-1.7b.yaml"

with open(data_path, "r", encoding="utf-8") as f:
    test_set = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, legacy=False
)
hparams = ROMEHyperParams.from_hparams(rome_path)

total = len(test_set)
success = 0
rephrase_success = 0
locality_success = 0

i = 1
for data in test_set:
    print(f"[{i}] 开始编辑 {data['prompt']} {data['ground_truth']} => {data['target_new']}")
    response = edit_by_rome(hparams, data, tokenizer)
    print(f"[{i}] 正在验证结果")
    print(f"[{i}] Prompt: {data['prompt']}, Response: {response[0]}")
    print(f"[{i}] Rephrase_Prompt: {data['rephrase_prompt']}, Response: {response[1]}")
    print(f"[{i}] Locality_Prompt: {data['locality_prompt']}, Response: {response[2]}")
    if data["target_new"] in response[0]:
        success += 1
    if data["target_new"] in response[1]:
        rephrase_success += 1
    if data["locality_ground_truth"] in response[2]:
        locality_success += 1
    i += 1

print("Efficacy Success: ", success / total)
print("Generalization Success: ", rephrase_success / total)
print("Locality Success: ", locality_success / total)
