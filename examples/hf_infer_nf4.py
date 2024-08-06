import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llmtask import TaskGenerator


dataset = "mmlu"
max_shot = 5
quant_type = "bf16"

model_id = "/hub/weights/LLM-Research/Meta-Llama-3___1-8B"

def log(msg):
    with open(f"{dataset}-{max_shot}shot_{quant_type}_nf4_gs64.log", "a") as f:
        f.write(f"{msg}\n")

device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map={"": 0},
    quantization_config=bnb_config
)

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
            do_sample=False,
            temperature=None,
            top_p=None, # UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
        )
    ans = tokenizer.batch_decode([generated_ids[0][input_tokens:]])[0]
    log(f"[{cnt:5}] [{(time.time() - t0):5.3f} s] => ans:{ans}")
    cnt += 1
    TG.feedback(ans)
    log(TG.summary())
    torch.cuda.empty_cache()
