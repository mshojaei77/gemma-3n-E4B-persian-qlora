---
license: gemma
language:
- fa
base_model: unsloth/gemma-3n-E4B-it
tags:
- gemma-3n
- unsloth
- persian
- farsi
- conversational
- qlora
- fine-tuned
- chat
- instruction-following
datasets:
- mshojaei77/persian-gk
model-index:
- name: gemma-3n-E4B-persian
  results: []
pipeline_tag: text-generation
widget:
- example_title: "Persian History Question"
  text: |
    <start_of_turn>user
    سلام! لطفاً در مورد تاریخ ایران توضیح کوتاهی بدهید.<end_of_turn>
    <start_of_turn>model
    سلام! تاریخ ایران بسیار غنی و طولانی است. ایران یکی از قدیمی‌ترین تمدن‌های جهان محسوب می‌شود که بیش از ۲۵۰۰ سال قدمت دارد. از امپراتوری هخامنشی که توسط کوروش کبیر بنیان گذاشته شد تا دوران صفویه، قاجار و پهلوی، ایران همواره نقش مهمی در تاریخ جهان داشته است.<end_of_turn>
- example_title: "Persian Culture Question"
  text: |
    <start_of_turn>user
    باغ تخت چه ویژگی‌هایی داره که اون رو به یکی از قدیمی‌ترین باغ‌های شیراز تبدیل کرده؟<end_of_turn>
    <start_of_turn>model
    باغ تخت در شیراز یکی از باغ‌های تاریخی و زیبای ایران است که ویژگی‌های منحصر به فردی دارد. این باغ با معماری سنتی ایرانی، آب‌نماهای زیبا، درختان کهنسال و فضای سبز گسترده شناخته می‌شود. طراحی چهارباغی آن و استفاده از عناصر طبیعی آن را به یکی از نمونه‌های برجسته باغ‌سازی ایرانی تبدیل کرده است.<end_of_turn>
- example_title: "General Knowledge Question"
  text: |
    <start_of_turn>user
    برنامه‌نویسی پایتون چیست و چه کاربردهایی دارد؟<end_of_turn>
    <start_of_turn>model
    پایتون یک زبان برنامه‌نویسی قدرتمند و آسان است که در سال ۱۹۹۱ توسط گیدو ون روسوم ایجاد شد. این زبان کاربردهای گسترده‌ای دارد از جمله: توسعه وب، هوش مصنوعی و یادگیری ماشین، تحلیل داده، اتوماسیون، توسعه بازی و برنامه‌های دسکتاپ. سادگی نحو و کتابخانه‌های غنی آن، پایتون را به انتخاب اول بسیاری از برنامه‌نویسان تبدیل کرده است.<end_of_turn>
---

# Gemma-3N 4B Persian - General Knowledge

<div align="center">
  <img src="https://github.com/user-attachments/assets/0c52d460-1831-46aa-b3e6-b1a5249c0174" alt="Hugging Face" width="500"/>
  <br>
  <strong>🇮🇷 Persian Language Model | 🤖 Conversational AI | 📚 General Knowledge</strong>
</div>

## Model Description

This model is a fine-tuned version of `unsloth/gemma-3n-E4B-it`, optimized for Persian (Farsi) conversational tasks focused on general knowledge. It employs QLoRA techniques for efficient adaptation and is merged into a standalone model suitable for deployment.

## Model Details

### Base Model and Architecture
- **Base Model**: `unsloth/gemma-3n-E4B-it` (Google Gemma 3N 4B Instruction-Tuned).
- **Model Type**: Causal language model.
- **Model Size**: Approximately 9.9 GB (16-bit precision).
- **Context Length**: Supports up to 32,768 tokens, trained with 4,000 tokens.
- **Vocabulary**: Gemma tokenizer vocabulary.

### Intended Uses
This model is designed for direct use in Persian conversational AI, including instruction-following and general knowledge queries in domains such as Persian heritage, programming, architecture, and tourism. It is suitable for downstream applications like chat interfaces or educational tools. Out-of-scope uses include non-Persian languages or safety-critical applications.

## Training

### Training Data
- **Dataset**: `mshojaei77/persian-gk` (cleaned version: `mshojaei77/persian-gk-cleaned`), comprising 5,897 Persian conversations in ChatML format.
- **Domains**: Programming, Persian heritage, architecture, tourism, and general Q&A.
- **License**: CC-BY-4.0.

### Training Procedure
The model was fine-tuned using QLoRA with 4-bit quantization.
- **LoRA Parameters**: Rank=8, alpha=16, dropout=0.0; target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Hyperparameters**: Learning rate=2e-5, batch size=2 (effective=8 with gradient accumulation=4), epochs=1, optimizer=AdamW 8-bit, weight decay=0.01, warmup steps=10, linear LR scheduler, seed=3407.
- **Framework**: Unsloth with Weights & Biases monitoring.
- **Infrastructure**: Google Colab with GPU acceleration.

The merging process integrated LoRA adapters into the base model, converting to 16-bit precision for standalone use.

## Bias, Risks, and Limitations

- **Limitations**: Optimized for Persian; performance may degrade in other languages. Knowledge is limited to training data (no real-time updates). Potential for hallucinations or biases from the dataset.
- **Risks**: May reflect cultural biases; not suitable for critical decisions (e.g., medical, legal).
- **Ethical Considerations**: Users should verify outputs, avoid sensitive data, and ensure ethical use.

## Usage

### Quick Start with Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mshojaei77/gemma-3n-E4B-persian"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "سلام! در مورد تاریخ ایران توضیح بدهید."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
```

Recommended parameters: `max_new_tokens=256-512`, `temperature=0.1-0.7`, `top_p=0.9-0.95`.

For memory optimization, use 8-bit quantization:

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
```

## Reproduction

Reproduce training using the provided notebooks: `training_qlora-4bit-colab-notebook.md` for QLoRA setup and `merging-adaptors-kaggle-notebook.md` for adapter merging.

## Related Resources

- **Base Model**: `unsloth/gemma-3n-E4B-it`.
- **Adapters**: `mshojaei77/gemma-3n-E4B-persian-lora-adapters`.
- **Dataset**: `mshojaei77/persian-gk`.
- **GitHub**: `mshojaei77/gemma-3n-E4B-persin-qlora`.
- **Frameworks**: Unsloth (arXiv:2305.14314), PEFT (arXiv:2106.09685), Transformers.

## Citation

```bibtex
@misc{gemma3n_persian_2024,
  title={Gemma-3N 4B Persian Fine-tuned Model},
  author={Shojaei, M.},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/mshojaei77/gemma-3n-E4B-persian},
  note={Fine-tuned using QLoRA on Persian General Knowledge dataset}
}
```

Dataset citation:

```bibtex
@misc{persian_gk_2024,
  title={persian-gk: Persian General Knowledge Chat Dataset},
  author={Shojaei, M. and Contributors},
  year={2024},
  url={https://huggingface.co/datasets/mshojaei77/persian-gk}
}
```

## License

Licensed under the Gemma Terms of Use (https://ai.google.dev/gemma/terms). Downstream users must adhere to these terms.

## Acknowledgments

Thanks to Google for the Gemma architecture, the Unsloth team for training tools, Hugging Face for hosting, and the Persian NLP community for contributions.