# Gemma-3N 4B Persian Fine-Tuning Project

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/mshojaei77/gemma-3n-E4B-persian)

## Project Overview

This repository provides a reproducible pipeline for fine-tuning Google's Gemma-3N 4B model for Persian (Farsi) conversational tasks using QLoRA. The result is a memory-efficient model optimized for general knowledge in Persian domains.

For detailed technical information, see the [technical_report.md](https://github.com/mshojaei77/gemma-3n-E4B-persian-qlora/blob/main/technical_report.md).

## Repository Structure

- `charts/`: Training visualizations.
- `training_qlora-4bit-colab-notebook.md`: Training pipeline.
- `merging-adaptors-kaggle-notebook.md`: Adapter merging process.
- `dataset-card.md`: Dataset details.
- `model-card.md`: Model documentation.
- `README.md`: This file.

## Installation

To reproduce the project, install the required dependencies:

```bash
pip install -q -U unsloth transformers peft bitsandbytes accelerate wandb datasets
```

## Reproducibility

To reproduce the fine-tuning:

1. Clone the repository:

```bash
git clone https://github.com/mshojaei77/gemma-3n-E4B-persian-qlora.git
cd gemma-3n-E4B-persian-qlora
```

2. Load and configure the model:

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    dtype=None,
    max_seq_length=4000,
    load_in_4bit=True,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=False,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
```

3. Load and preprocess the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("mshojaei77/persian-gk", split="train")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
def formatting_prompts_func(examples):
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in examples["messages"]]
    return {"text": texts}
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda example: len(example.get("text", "").strip()) > 0)
```

4. Configure and run the trainer:

```python
from trl import SFTTrainer
from trl import SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=10,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        seed=3407,
        report_to="wandb",
    ),
)
trainer.train()
```

5. Merge and save:
Follow the [merging notebook](./merging-adaptors-kaggle-notebook.md) for detailed code.

## Datasets

- **[persian-gk](https://huggingface.co/datasets/mshojaei77/persian-gk)**: 5,897 Persian conversations in ChatML format.
- Cleaned version: **[persian-gk-cleaned](https://huggingface.co/datasets/mshojaei77/persian-gk-cleaned)**.

## Models

- Adapters: **[gemma-3n-E4B-persian-lora-adapters](https://huggingface.co/mshojaei77/gemma-3n-E4B-persian-lora-adapters)**.
- Merged Model: **[gemma-3n-E4B-persian](https://huggingface.co/mshojaei77/gemma-3n-E4B-persian)**.

## Technical Specifications

- Base Model: `unsloth/gemma-3n-E4B-it`.
- Technique: QLoRA (4-bit quantization, rank=8, alpha=16).
- Hyperparameters: Learning rate=2e-5, effective batch size=8, epochs=1.
- Memory: Training ~11.5GB peak on T4 GPU, inference ~10GB in FP16.

## Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mshojaei77/gemma-3n-E4B-persian"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "سلام! در مورد تاریخ ایران توضیح بدهید."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=512, temperature=0.7, top_p=0.95, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Use Cases

- Persian conversational AI.
- Educational tools for Persian heritage and programming.

## Limitations & Safety

- **Language Scope**: Optimised for Persian (Farsi). Accuracy and fluency drop in other languages.
- **Knowledge Cut-off**: Training data ends at January 2024; the model is unaware of events after this date.
- **Hallucination**: May generate plausible-sounding but incorrect or fabricated answers—always verify critical outputs.
- **Domain Transfer**: Performance declines on highly specialised or safety-critical topics (medical, legal, financial).
- **Context Window**: Training never saw prompts longer than 4 k tokens; very long inputs can degrade quality.
- **Resource Needs**: FP16 inference requires ≈ 10 GB GPU VRAM; use 8-bit/4-bit quantisation for lower-resource devices.

### Ethical Considerations

- Outputs can reflect cultural or societal biases present in the source data.
- Do **not** rely on the model as the sole source of professional advice.
- Deploy with content filters and human oversight, especially for minors or vulnerable users.
- Ensure compliance with Gemma Terms of Use, dataset licence (CC-BY-4.0), and privacy regulations.

## License

- Model: Gemma Terms of Use.
- Dataset: CC-BY-4.0.
- Code: MIT.

## Acknowledgments

Thanks to Google, Unsloth, Hugging Face, and the Persian NLP community.