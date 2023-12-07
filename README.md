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



```bash
root@xxx:~# python test.py 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [00:02<00:00, 516.32it/s]
{'Avg': 0.24755062050947094, 'STEM': 0.21495327102803738, 'other': 0.22535211267605634, 'social sciences': 0.22255192878338279, 'humanities': 0.29922779922779924}
```
