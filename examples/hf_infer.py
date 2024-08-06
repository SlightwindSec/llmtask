import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmtask import TaskGenerator

dataset = "mmlu"
max_shot = 5
quant_type = "bf16"

model_id = "/hub/weights/LLM-Research/Meta-Llama-3___1-8B"

def log(msg):
    with open(f"{dataset}-{max_shot}shot_{quant_type}.log", "a") as f:
        f.write(f"{msg}\n")

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id)


TG = TaskGenerator(dataset, max_shot=max_shot)
cnt = 0
for task in TG:
    model_inputs = tokenizer([task], return_tensors="pt").to(device)
    input_tokens = len(model_inputs['input_ids'][0])
    t0 = time.time()
    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    ans = tokenizer.batch_decode([generated_ids[0][input_tokens:]])[0]
    log(f"[{cnt:5}] [{(time.time() - t0):5.3f} s] => ans:{ans}")
    cnt += 1
    TG.feedback(ans)
    log(TG.summary())
    torch.cuda.empty_cache()
