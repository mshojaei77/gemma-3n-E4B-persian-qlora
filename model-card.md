---
license: apache-2.0
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
- 4bit
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
- example_title: "Persian Culture Question"
  text: |
    <start_of_turn>user
    باغ تخت چه ویژگی‌هایی داره که اون رو به یکی از قدیمی‌ترین باغ‌های شیراز تبدیل کرده؟<end_of_turn>
    <start_of_turn>model
- example_title: "General Knowledge Question"
  text: |
    <start_of_turn>user
    برنامه‌نویسی پایتون چیست و چه کاربردهایی دارد؟<end_of_turn>
    <start_of_turn>model
---

# Gemma-3N 4B - Persian General Knowledge (Fine-tuned & Merged)

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" alt="Hugging Face" width="100"/>
  <br>
  <strong>🇮🇷 Persian Language Model | 🤖 Conversational AI | 📚 General Knowledge</strong>
</div>

## Model Overview

This model is a fine-tuned version of `unsloth/gemma-3n-E4B-it`, specifically optimized for **Persian (Farsi)** conversational general knowledge tasks. The model has been trained using state-of-the-art QLoRA (Quantized Low-Rank Adaptation) techniques and subsequently merged into a standalone deployment-ready model.

### Key Features

- 🎯 **Specialized for Persian**: Optimized for Farsi language understanding and generation
- 💬 **Conversational AI**: Designed for chat-style interactions and instruction following
- 📖 **General Knowledge**: Covers diverse topics including Persian heritage, programming, architecture, and tourism
- ⚡ **Memory Efficient**: Trained using 4-bit quantization with QLoRA
- 🚀 **Production Ready**: Merged model without adapter dependencies
- 📊 **Monitored Training**: Complete training metrics and visualization available

## Model Details

### Architecture
- **Base Model**: `unsloth/gemma-3n-E4B-it` (Google Gemma 3N 4B Instruction-Tuned)
- **Model Type**: Causal Language Model
- **Model Size**: ~8.5 GB (16-bit precision)
- **Context Length**: 4,000 tokens
- **Vocabulary Size**: Gemma tokenizer vocabulary

### Training Details

#### Dataset
- **Dataset**: [`mshojaei77/persian-gk`](https://huggingface.co/datasets/mshojaei77/persian-gk)
- **Size**: 5,897 Persian conversations
- **Format**: ChatML-style conversations with system, user, and assistant roles
- **Domains**: Programming, Persian heritage, architecture, tourism, general Q&A
- **License**: CC-BY-4.0

#### Fine-tuning Configuration

**QLoRA Parameters:**
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Base Precision**: 4-bit quantization
- **LoRA Rank (r)**: 8
- **LoRA Alpha**: 16 (scaling factor: 2.0)
- **LoRA Dropout**: 0.0
- **Target Modules**: MLP layers (attention modules disabled for memory efficiency)
- **Bias Strategy**: None

**Training Hyperparameters:**
- **Learning Rate**: 2e-5
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps (effective batch size: 8)
- **Epochs**: 1
- **Optimizer**: AdamW 8-bit
- **Weight Decay**: 0.01
- **Warmup Steps**: 10
- **LR Scheduler**: Linear decay
- **Seed**: 3407

**Memory Optimization:**
- **Vision Layers**: Disabled (text-only training)
- **Language Layers**: Enabled
- **Attention Modules**: Disabled (memory efficient)
- **MLP Modules**: Enabled (critical for adaptation)

#### Training Infrastructure
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth)
- **Monitoring**: Weights & Biases integration
- **Environment**: Google Colab with GPU acceleration
- **Memory Usage**: Optimized for limited GPU memory environments

#### Training Metrics

The model training was monitored with comprehensive metrics:

- **Training Loss**: Tracked throughout the training process
- **Gradient Norm**: Monitored for training stability
- **Memory Usage**: Efficient 4-bit training with <12GB GPU memory
- **Training Time**: Optimized for fast iteration

*Training charts available in the repository: `train-loss Chart` and `train-grade_norm Chart`*

### Merging Process

The final model underwent a sophisticated merging process to create a standalone deployment model:

1. **Base Model Loading**: Loaded `unsloth/gemma-3n-E4B-it` in 4-bit precision
2. **Adapter Application**: Applied trained LoRA adapters from `mshojaei77/gemma-3n-E4B-persin-lora-adaptors`
3. **Mathematical Merging**: Integrated adapter weights using the formula:
   ```
   W_merged = W_base + (LoRA_A × LoRA_B) × (alpha/r)
   ```
4. **De-quantization**: Converted from 4-bit to 16-bit precision
5. **Standalone Model**: Removed adapter dependencies for direct deployment

## Usage

### Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "mshojaei77/gemma-3n-E4B-persian"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prepare conversation
messages = [{
    "role": "user",
    "content": "سلام! در مورد تاریخ ایران توضیح بدهید."
}]

# Tokenize input
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Advanced Usage

#### Streaming Generation

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(
    inputs,
    max_new_tokens=500,
    temperature=0.1,
    top_p=0.95,
    top_k=64,
    streamer=streamer,
    do_sample=True,
)
```

#### Batch Processing

```python
# Multiple conversations
conversations = [
    [{"role": "user", "content": "برنامه‌نویسی پایتون چیست؟"}],
    [{"role": "user", "content": "تاریخ ایران را توضیح دهید."}]
]

# Process in batch
inputs = tokenizer.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
```

### Generation Parameters

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `max_new_tokens` | 256-512 | Maximum response length |
| `temperature` | 0.1-0.7 | Creativity control (lower = more focused) |
| `top_p` | 0.9-0.95 | Nucleus sampling threshold |
| `top_k` | 50-64 | Top-k sampling limit |
| `do_sample` | True | Enable sampling for diverse responses |

## Performance

### Model Specifications
- **Model Size**: ~8.5 GB (16-bit precision)
- **Memory Requirements**: 
  - **Inference**: 12+ GB GPU memory
  - **Training**: 8+ GB GPU memory (with 4-bit quantization)
- **Context Window**: 4,000 tokens
- **Language**: Optimized for Persian (Farsi)
- **Domain**: General knowledge and conversational AI

### Benchmarks

*Note: Formal benchmarks on Persian language tasks are planned for future releases.*

**Qualitative Assessment:**
- ✅ Strong performance on Persian cultural and historical questions
- ✅ Good programming knowledge in Persian context
- ✅ Coherent conversational abilities
- ✅ Appropriate cultural context understanding

## Limitations and Considerations

### Known Limitations
- **Language Specialization**: Primarily optimized for Persian; may have reduced performance in other languages
- **Knowledge Cutoff**: Limited to training data knowledge (no real-time information)
- **Hallucination Risk**: May generate plausible but incorrect information
- **Cultural Bias**: May reflect biases present in the training dataset
- **Context Length**: Limited to 4,000 tokens per conversation

### Responsible AI Considerations
- **Not for Critical Applications**: Should not be used for medical, legal, or safety-critical decisions
- **Human Oversight Required**: Outputs should be reviewed by humans for important applications
- **Bias Awareness**: Users should be aware of potential cultural and linguistic biases
- **Privacy**: Do not input sensitive personal information

### Ethical Guidelines
- Use responsibly and ethically
- Respect cultural sensitivities
- Verify important information from authoritative sources
- Consider the impact of generated content

## Technical Implementation

### Memory Optimization

```python
# For memory-constrained environments
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Enable gradient checkpointing for training
model.gradient_checkpointing_enable()
```

### Quantization for Deployment

```python
# 8-bit quantization for reduced memory
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
)
```

## Training Reproduction

To reproduce the training process:

1. **Setup Environment**: Follow the installation steps in `training_qlora-4bit-colab-notebook.md`
2. **Load Dataset**: Use `mshojaei77/persian-gk` dataset
3. **Configure Training**: Apply the QLoRA configuration as documented
4. **Monitor Training**: Use Weights & Biases for tracking
5. **Merge Adapters**: Follow the merging process in `merging-adaptors-kaggle-notebook.md`

### Training Files
- [`training_qlora-4bit-colab-notebook.md`](./training_qlora-4bit-colab-notebook.md) - Complete training pipeline
- [`merging-adaptors-kaggle-notebook.md`](./merging-adaptors-kaggle-notebook.md) - Adapter merging process
- [`dataset-card.md`](./dataset-card.md) - Dataset documentation
- [`charts/`](./charts/) - Training visualization charts

## Related Resources

### Models
- **Base Model**: [`unsloth/gemma-3n-E4B-it`](https://huggingface.co/unsloth/gemma-3n-E4B-it)
- **LoRA Adapters**: [`mshojaei77/gemma-3n-E4B-persin-lora-adaptors`](https://huggingface.co/mshojaei77/gemma-3n-E4B-persin-lora-adaptors)

### Datasets
- **Training Dataset**: [`mshojaei77/persian-gk`](https://huggingface.co/datasets/mshojaei77/persian-gk)

### Frameworks and Tools
- **Unsloth**: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **PEFT**: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- **Transformers**: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

### Research Papers
- **QLoRA**: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- **Gemma**: [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- **LoRA**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

## Citation

If you use this model in your research or applications, please cite:

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

### Dataset Citation
```bibtex
@misc{persian_gk_2024,
  title={persian-gk: Persian General Knowledge Chat Dataset},
  author={Shojaei, M. and Contributors},
  year={2024},
  url={https://huggingface.co/datasets/mshojaei77/persian-gk}
}
```

## License

This model is licensed under the **Apache License 2.0**. The training dataset is licensed under **CC-BY-4.0**.

## Acknowledgments

- **Google** for the Gemma model architecture
- **Unsloth Team** for the efficient training framework
- **Hugging Face** for the model hosting and ecosystem
- **Persian NLP Community** for dataset contributions and feedback

---

<div align="center">
  <strong>Built with ❤️ for the Persian NLP community</strong>
  <br>
  <em>Contributions and feedback are welcome!</em>
</div>