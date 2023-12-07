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
