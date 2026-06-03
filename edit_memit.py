import sys
import os
import json
import time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
easyedit_root = r"./EasyEdit"
if easyedit_root not in sys.path:
    sys.path.append(easyedit_root)
from easyeditor import BaseEditor, MEMITHyperParams

model_path = "Qwen/Qwen3-1.7B"  # 保持与 yaml 中一致
data_path = "./zsre_data.json"
memit_path = "./memit/qwen3-1.7b.yaml"


with open(data_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

datas = raw[:500]
prompts = [data["src"] for data in datas]
rephrase_prompts = [data["rephrase"] for data in datas]
target_new = [data["alt"] for data in datas]
subject = [data["subject"] for data in datas]
locality_inputs = {
    "neighborhood": {
        "prompt": [data["loc"] for data in datas],
        "ground_truth": [data["loc_ans"] for data in datas],
    },
}

hparams = MEMITHyperParams.from_hparams(memit_path)
editor = BaseEditor.from_hparams(hparams)

print("-- 开始MEMIT知识批量编辑--")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.perf_counter()

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    rephrase_prompts=rephrase_prompts,
    target_new=target_new,
    subject=subject,
    locality_inputs=locality_inputs,
    keep_original_weight=True,
)

torch.cuda.synchronize()
end_time = time.perf_counter()
print("-- MEMIT知识批量编辑结束--")
elapsed_time = end_time - start_time
max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
max_reserv = torch.cuda.max_memory_reserved() / (1024**2)
print("===== MEMIT 编辑统计结果 =====")
print(f"总耗时 (Time Cost): {elapsed_time:.1f} 秒")
print(f"实际显存峰值 (Max Allocated VRAM): {max_alloc:.1f} MB ({max_alloc / 1024:.1f} GB)")
print(f"预留显存峰值 (Max Reserved VRAM): {max_reserv:.1f} MB ({max_reserv / 1024:.1f} GB)")
