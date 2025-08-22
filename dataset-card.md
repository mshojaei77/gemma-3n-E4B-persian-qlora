---
dataset_info:
  features:
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  splits:
  - name: train
    num_bytes: 7201128.0
    num_examples: 5897
  download_size: 2969604
  dataset_size: 7201128.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-4.0
task_categories:
- question-answering
- text-generation
language:
- fa
tags:
- persian
- farsi
pretty_name: Persian General Knowledge
---

# Dataset Card for persian-gk (Persian General Knowledge)

## Dataset Summary
`persian-gk` is a cleaned and structured collection of Persian (Farsi) conversation pairs covering a wide range of general-knowledge topics.  Each conversation is formatted in [ChatML](https://github.com/openai/openai-python) style with explicit *system*, *user*, and *assistant* roles, enabling straightforward use for both instruction-tuning and chat-style language-model training.

* **Language:** Persian (fa)
* **Size:** 5 897 conversations, 2–8 turns each (≈ 150 000 message lines)
* **Domains:** programming, Persian heritage, architecture, tourism, and assorted Q&A,...
* **License:** CC-BY-4.0
* **Source:** Curated from public Persian blogs, Q&A resources, and manually written system prompts.

## Supported Tasks and Benchmarks
1. **Instruction Tuning / Chat Completion** – Fine-tune models for Persian dialogue or QA.
2. **Knowledge-Grounded Generation** – Evaluate a model’s factual consistency in Persian.
3. **Domain Adaptation** – Adapt multilingual models to Persian general-knowledge domains.

_No public benchmark results are yet reported._

## Data Splits
The current release provides a single `train` split (5 897 examples).  Future versions may introduce `validation` and `test` splits.

| Split | Examples |
|-------|----------|
| train | 5 897    |

## Usage
```python
from datasets import load_dataset

ds = load_dataset("mshojaei77/persian-gk", split="train")
print(ds[0]["messages"])
```

### ChatML to plain prompt
```python
def format_chatml(example):
    return "\n".join(f"<{m['role']}> {m['content']}" for m in example["messages"])
```

## Citation
If you use this dataset, please cite:
```
@misc{persian_gk_2024,
  title  = {persian-gk: Persian General Knowledge Chat Dataset},
  author = {Shojaei, M. and Contributors},
  year   = {2024},
  url    = {https://huggingface.co/datasets/mshojaei77/persian-gk}
}
```

## License
Licensed under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license.