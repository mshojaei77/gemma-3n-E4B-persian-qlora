# Gemma-3N 4B Persian Fine-tuning Project

<div align="center">
  <img src="https://github.com/user-attachments/assets/0c52d460-1831-46aa-b3e6-b1a5249c0174" alt="Hugging Face" width="100"/>
  <br>
  <strong>🇮🇷 Persian Language Model | 🤖 Conversational AI | 📚 General Knowledge</strong>
</div>

## 📋 Project Overview

This repository contains the complete pipeline for fine-tuning Google's Gemma-3N 4B model for Persian (Farsi) language tasks using QLoRA (Quantized Low-Rank Adaptation). The project includes training notebooks, dataset preparation, model merging, and deployment resources.

### 🎯 Key Achievements

- ✅ Successfully fine-tuned Gemma-3N 4B for Persian conversations
- ✅ Implemented memory-efficient QLoRA training (4-bit quantization)
- ✅ Created comprehensive Persian general knowledge dataset
- ✅ Merged adapters into standalone deployment model
- ✅ Achieved production-ready Persian conversational AI

## 🗂️ Repository Structure

```
gemma-3n-E4B-persin-qlora/
├── 📊 charts/                              # Training visualization charts
│   ├── train-loss Chart 8_22_2025, 8_44_13 PM.png
│   └── train-grade_norm Chart 8_22_2025, 9_49_25 PM.png
├── 📚 docs/                               # Additional documentation
├── 🔧 training_qlora-4bit-colab-notebook.md    # Complete training pipeline
├── 🔗 merging-adaptors-kaggle-notebook.md      # Adapter merging process
├── 📄 dataset-card.md                    # Dataset documentation
├── 🏷️ model-card.md                      # Comprehensive model card
└── 📖 README.md                          # This file
```

## 🚀 Quick Start

### 1. Training the Model

Follow the complete training pipeline:

📓 **[Training Notebook](./training_qlora-4bit-colab-notebook.md)**
- Environment setup and dependencies
- Model and tokenizer configuration
- QLoRA adapter setup
- Dataset loading and preprocessing
- Training execution with monitoring
- Memory optimization techniques

### 2. Merging Adapters

Convert trained adapters to standalone model:

🔗 **[Merging Notebook](./merging-adaptors-kaggle-notebook.md)**
- Base model loading
- Adapter application
- Mathematical weight merging
- Model validation and testing
- Hugging Face Hub deployment

## 📊 Datasets

### Primary Training Dataset
🔗 **[persian-gk](https://huggingface.co/datasets/mshojaei77/persian-gk)**
- 5,897 Persian conversation pairs
- ChatML format for instruction tuning
- Domains: Programming, Persian heritage, architecture, tourism
- License: CC-BY-4.0

### Cleaned Dataset Version
🔗 **[persian-gk-cleaned](https://huggingface.co/datasets/mshojaei77/persian-gk-cleaned)**
- Enhanced version with improved quality
- Additional preprocessing and validation
- Optimized for training efficiency

## 🤖 Models

### LoRA Adapters
🔗 **[gemma-3n-E4B-persin-lora-adaptors](https://huggingface.co/mshojaei77/gemma-3n-E4B-persin-lora-adaptors)**
- Trained QLoRA adapters
- 4-bit quantized training
- Rank 8, Alpha 16 configuration
- Memory-efficient MLP targeting

### Final Merged Model
🔗 **[gemma-3n-E4B-persian](https://huggingface.co/mshojaei77/gemma-3n-E4B-persian)**
- Production-ready standalone model
- No adapter dependencies
- 16-bit precision for deployment
- Optimized for Persian conversations

## 🛠️ Technical Specifications

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | `unsloth/gemma-3n-E4B-it` | Google Gemma 3N 4B Instruction-Tuned |
| **Technique** | QLoRA | Quantized Low-Rank Adaptation |
| **Quantization** | 4-bit | Memory-efficient training |
| **LoRA Rank** | 8 | Adaptation rank |
| **LoRA Alpha** | 16 | Scaling factor |
| **Learning Rate** | 2e-5 | Optimized for stability |
| **Batch Size** | 2 (effective: 8) | With gradient accumulation |
| **Epochs** | 1 | Single pass training |

### Memory Requirements

- **Training**: 8+ GB GPU memory (with 4-bit quantization)
- **Inference**: 12+ GB GPU memory (16-bit model)
- **Context Length**: 4,000 tokens
- **Model Size**: ~8.5 GB (merged model)

## 📈 Training Results

The training process was comprehensively monitored with:

- **Loss Tracking**: Consistent convergence throughout training
- **Gradient Monitoring**: Stable gradient norms
- **Memory Efficiency**: <12GB GPU usage during training
- **Performance**: Strong Persian conversation capabilities

*Detailed charts available in the [`charts/`](./charts/) directory*

## 🔧 Usage Examples

### Basic Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the merged model
model_id = "mshojaei77/gemma-3n-E4B-persian"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Persian conversation
messages = [{
    "role": "user",
    "content": "سلام! در مورد تاریخ ایران توضیح بدهید."
}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using LoRA Adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3n-E4B-it",
    load_in_4bit=True
)

# Apply LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "mshojaei77/gemma-3n-E4B-persin-lora-adaptors"
)
```

## 📚 Documentation

### Core Documentation
- 📄 **[Model Card](./model-card.md)** - Comprehensive model documentation
- 📄 **[Dataset Card](./dataset-card.md)** - Dataset details and specifications
- 📓 **[Training Notebook](./training_qlora-4bit-colab-notebook.md)** - Complete training pipeline
- 🔗 **[Merging Notebook](./merging-adaptors-kaggle-notebook.md)** - Adapter merging process

### Additional Resources
- 📊 **[Training Charts](./charts/)** - Loss and gradient visualizations
- 📚 **[Documentation](./docs/)** - Extended documentation and guides

## 🎯 Use Cases

### Primary Applications
- **Persian Conversational AI**: Chat applications and virtual assistants
- **Educational Tools**: Persian language learning and tutoring
- **Content Generation**: Persian text generation for various domains
- **Cultural Preservation**: Persian heritage and cultural knowledge sharing

### Domain Expertise
- Programming and technology (in Persian)
- Persian history and culture
- Architecture and tourism
- General knowledge Q&A

## ⚠️ Limitations

- **Language Focus**: Optimized primarily for Persian; reduced performance in other languages
- **Knowledge Cutoff**: Limited to training data knowledge
- **Context Length**: 4,000 token limit per conversation
- **Hallucination Risk**: May generate plausible but incorrect information
- **Cultural Bias**: May reflect biases present in training data

## 🤝 Contributing

Contributions are welcome! Please consider:

- **Dataset Improvements**: Additional Persian conversation data
- **Evaluation Metrics**: Persian language benchmarks
- **Documentation**: Enhanced guides and examples
- **Bug Reports**: Issues with training or inference

## 📄 License

- **Model**: Apache License 2.0
- **Dataset**: CC-BY-4.0
- **Code**: MIT License (repository code)

## 🙏 Acknowledgments

- **Google** for the Gemma model architecture
- **Unsloth Team** for efficient training framework
- **Hugging Face** for model hosting and ecosystem
- **Persian NLP Community** for dataset contributions

## 📞 Contact

For questions, suggestions, or collaborations:

- **Hugging Face**: [@mshojaei77](https://huggingface.co/mshojaei77)
- **Issues**: Use GitHub Issues for bug reports and feature requests

---

<div align="center">
  <strong>🇮🇷 Built with ❤️ for the Persian NLP community</strong>
  <br>
  <em>Star ⭐ this repository if you find it useful!</em>
</div>
