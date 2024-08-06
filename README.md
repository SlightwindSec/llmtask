# LLM Task

## Introduction



## Quick Start

```bash
pip install llmtask
```


```python
import random

from llmtask import TaskGenerator

choices = ("A", "B", "C", "D")

TG = TaskGenerator("mmlu", max_shot=4)

for task in TG:
    TG.feedback(random.choice(choices))

print(TG.summary())
```



| Model        | Dataset            | Avg (%) | Quant     | Group Size | Time (s) |
| ------------ | ------------------ | ------- | --------- | ---------- | -------- |
| Llama-3.1-8b | CEval-val (5-shot) | 52.97   | BF16      | N/A        | 0.1664   |
| Llama-3.1-8b | CEval-val (5-shot) | 49.70   | HQQ 4-bit | 64         | 0.3210   |
| Llama-3.1-8b | CEval-val (5-shot) | 50.96   | NF4       | 64         | 0.2023   |
| Llama-3.1-8b | MMLU-val (5-shot)  | 63.75   | BF16      | N/A        | 0.1926   |
| Llama-3.1-8b | MMLU-val (5-shot)  | 61.59   | HQQ 4-bit | 64         | 0.3538   |
| Llama-3.1-8b | MMLU-val (5-shot)  | 61.85   | NF4       | 64         | 0.2246   |



