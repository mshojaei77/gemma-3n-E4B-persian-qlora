# Fine-tuning Gemma 3N (4B) for Persian Language using QLoRA

## Overview

This notebook demonstrates how to fine-tune the Gemma 3N (4B) model for Persian language tasks using QLoRA (Quantized Low-Rank Adaptation). The approach uses 4-bit quantization to reduce memory requirements while maintaining training effectiveness.

### Key Features:
- **Model**: Gemma 3N (4B) - Google's efficient language model
- **Technique**: QLoRA with 4-bit quantization
- **Dataset**: Persian General Knowledge dataset
- **Memory Optimization**: Designed for limited GPU memory environments
- **Monitoring**: Weights & Biases integration for training tracking

---

## 1. Environment Setup and Dependencies

### Install Required Packages

```python
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Colab-specific installation with version compatibility
    import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + "0.0.32.post2" if v == "2.8.0" else "0.0.29.post3"
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
```

### Install Gemma 3N Specific Dependencies

```python
%%capture
# Required specifically for Gemma 3N architecture
!pip install --no-deps --upgrade timm
```

### Import Required Libraries

```python
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
import torch
```

---

## 2. Model and Tokenizer Loading

### Load Pre-trained Model with 4-bit Quantization

```python
# Load the base Gemma 3N model with optimizations
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",  # Instruction-tuned variant
    dtype=None,                            # Auto-detect optimal dtype
    max_seq_length=4000,                   # Maximum sequence length for training
    load_in_4bit=True,                     # Enable 4-bit quantization for memory efficiency
)
```

**Configuration Explanation:**
- `model_name`: Uses the instruction-tuned version optimized for chat/instruction following
- `dtype=None`: Automatically selects the best data type (typically bfloat16)
- `max_seq_length=4000`: Balances memory usage with context length
- `load_in_4bit=True`: Reduces memory footprint by ~75% with minimal quality loss

---

## 3. LoRA Adapter Configuration

### Add QLoRA Adapters to the Model

```python
model = FastModel.get_peft_model(
    model,
    # Layer-specific fine-tuning configuration
    finetune_vision_layers=False,      # Disable vision layers (text-only training)
    finetune_language_layers=True,     # Enable language model layers
    finetune_attention_modules=False,  # Disable attention (better for GRPO, memory efficient)
    finetune_mlp_modules=True,         # Enable MLP layers (critical for adaptation)
    
    # LoRA hyperparameters
    r=8,                               # LoRA rank: controls adapter capacity
    lora_alpha=16,                     # LoRA scaling factor
    lora_dropout=0,                    # Dropout rate for LoRA layers
    bias="none",                       # Bias handling strategy
    random_state=3407,                 # Reproducibility seed
)
```

**LoRA Hyperparameter Details:**

- **`r=8` (LoRA Rank)**:
  - Controls the dimensionality of the low-rank adaptation
  - Higher values (16, 32) = more parameters, better adaptation, higher memory usage
  - Lower values (4, 8) = fewer parameters, faster training, potential underfitting
  - 8 is optimal for most tasks balancing quality and efficiency

- **`lora_alpha=16` (Scaling Factor)**:
  - Controls the magnitude of LoRA updates
  - Formula: `scaling = lora_alpha / r`
  - Common practice: `lora_alpha = 2 * r` or `lora_alpha = r`
  - Higher values = stronger adaptation, risk of overfitting

- **`lora_dropout=0` (Dropout Rate)**:
  - Regularization technique to prevent overfitting
  - 0.0 = no dropout (faster convergence)
  - 0.1-0.3 = moderate regularization
  - Use higher values (0.1+) for larger datasets

- **`bias="none"` (Bias Strategy)**:
  - "none": Don't adapt bias terms (memory efficient)
  - "all": Adapt all bias terms (more parameters)
  - "lora_only": Only adapt LoRA-specific biases

---

## 4. Dataset Loading and Preprocessing

### Load Persian Dataset

```python
# Load the Persian General Knowledge dataset
dataset = load_dataset("mshojaei77/persian-gk", split="train")
```

### Configure Chat Template

```python
# Apply Gemma 3 chat template for proper formatting
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",  # Use Gemma 3 specific chat format
)
```

### Data Formatting and Cleaning

```python
def formatting_prompts_func(examples):
    """Convert conversation format to training text format."""
    texts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        ).removeprefix('<bos>') 
        for convo in examples["messages"]
    ]
    return {"text": texts}

# Apply formatting and filter empty examples
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda example: len(example.get("text", "").strip()) > 0)

# Standardize data formats for consistency
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

# Verify data formatting
print("Sample formatted data:")
print(dataset[100]["text"])
```

---

## 5. Training Configuration and Monitoring

### Setup Weights & Biases Tracking

```python
!pip install --upgrade wandb

import wandb

# Login to Weights & Biases for experiment tracking
wandb.login()  # This will prompt for your API key
```

### Configure Training Parameters

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        # Data configuration
        dataset_text_field="text",
        
        # Batch size and gradient settings
        per_device_train_batch_size=2,      # Batch size per GPU
        gradient_accumulation_steps=4,       # Effective batch size = 2 * 4 = 8
        
        # Learning rate schedule
        learning_rate=2e-5,                  # Peak learning rate
        warmup_steps=10,                     # Learning rate warmup
        lr_scheduler_type="linear",          # Learning rate decay strategy
        
        # Training duration
        num_train_epochs=1,                  # Number of complete dataset passes
        
        # Optimization settings
        optim="adamw_8bit",                  # 8-bit AdamW optimizer
        weight_decay=0.01,                   # L2 regularization
        
        # Monitoring and reproducibility
        logging_steps=1,                     # Log every step
        seed=3407,                           # Random seed for reproducibility
        report_to="wandb",                   # Send metrics to Weights & Biases
    ),
)
```

**Training Hyperparameter Explanations:**

- **`per_device_train_batch_size=2`**:
  - Number of samples processed per GPU per step
  - Lower values = less memory usage, more stable gradients
  - Higher values = faster training, more memory usage
  - Adjust based on available GPU memory

- **`gradient_accumulation_steps=4`**:
  - Simulates larger batch sizes without memory overhead
  - Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps`
  - Higher values = more stable gradients, slower updates

- **`learning_rate=2e-5`**:
  - Peak learning rate for the optimizer
  - 2e-5 (0.00002) is optimal for most language model fine-tuning
  - Too high = unstable training, too low = slow convergence
  - For LoRA, can often use higher rates (5e-5) than full fine-tuning

- **`warmup_steps=10`**:
  - Gradually increases learning rate from 0 to peak over first N steps
  - Prevents early training instability
  - Typical range: 5-10% of total training steps

- **`weight_decay=0.01`**:
  - L2 regularization to prevent overfitting
  - 0.01 is standard for transformer models
  - Higher values = stronger regularization

- **`optim="adamw_8bit"`**:
  - 8-bit quantized AdamW optimizer
  - Reduces optimizer memory usage by ~50%
  - Minimal impact on training quality

### Verify Data Processing

```python
# Check tokenized input format
print("Tokenized sample:")
print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
```

---

## 6. Memory Monitoring

### Pre-training Memory Statistics

```python
# Monitor GPU memory usage before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Reserved memory before training: {start_gpu_memory} GB")
```

---

## 7. Model Training

### Execute Training Process

```python
# Start the fine-tuning process
# To resume from checkpoint: trainer.train(resume_from_checkpoint=True)
trainer_stats = trainer.train()
```

### Post-training Memory Analysis

```python
# Calculate memory usage and training statistics
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")
print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
print(f"Peak memory usage: {used_memory} GB ({used_percentage}% of total)")
print(f"Memory used for LoRA training: {used_memory_for_lora} GB ({lora_percentage}% of total)")
```

---

## 8. Model Saving

### Save LoRA Adapters

```python
# Define save location
model_save_name = "mshojaei77/gemma-3n-E4B-persin-lora-adaptors"

# Save only the LoRA adapters (not the full model)
model.save_pretrained(model_save_name)
tokenizer.save_pretrained(model_save_name)

print(f"LoRA adapters saved to: {model_save_name}")
print("Note: This saves only the adapter weights, not the full model.")
```

**Important Notes:**
- Only LoRA adapters are saved (~10-50MB vs several GB for full model)
- To use the model, you need both the base model and these adapters
- For deployment, consider merging adapters with base model

---

## 9. Model Testing and Inference

### Test the Fine-tuned Model

```python
# Prepare a test message in Persian
messages = [{
    "role": "user",
    "content": [{
        "type": "text", 
        "text": "باغ تخت چه ویژگی‌هایی داره که اون رو به یکی از قدیمی‌ترین باغ‌های شیراز تبدیل کرده؟"
    }]
}]

# Tokenize the input
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Add generation prompt for inference
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
).to("cuda")

# Generate response with streaming output
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    **inputs,
    max_new_tokens=500,          # Maximum tokens to generate
    temperature=0.1,             # Lower = more focused, higher = more creative
    top_p=0.95,                  # Nucleus sampling threshold
    top_k=64,                    # Top-k sampling limit
    streamer=streamer,           # Enable real-time output streaming
)
```

**Generation Parameters:**
- **`max_new_tokens=500`**: Maximum length of generated response
- **`temperature=0.1`**: Controls randomness (0.1 = focused, 1.0 = creative)
- **`top_p=0.95`**: Nucleus sampling - considers tokens with cumulative probability ≤ 0.95
- **`top_k=64`**: Only consider top 64 most likely tokens at each step

---

## 10. Next Steps and Limitations

### Current Limitations

⚠️ **Memory Constraints**: Due to RAM limitations in this environment, the adapter merging process needs to be completed in a separate environment (see `merging-adaptors-kaggle-notebook.md`).

### Recommended Next Steps

1. **Adapter Merging**: Combine LoRA adapters with base model for deployment
2. **Model Evaluation**: Test on validation dataset and benchmark tasks
3. **Quantization**: Convert to GGUF format for efficient inference
4. **Deployment**: Set up inference server or integrate into applications

### Performance Optimization Tips

- **Increase `r` value** (16, 32) for better adaptation quality
- **Adjust `lora_alpha`** to control adaptation strength
- **Use gradient checkpointing** for even lower memory usage
- **Experiment with different learning rates** (1e-5 to 5e-5)
- **Add validation dataset** for better training monitoring

---

## Resources and References

- **Unsloth Documentation**: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **QLoRA Paper**: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- **Gemma Model**: [https://huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b)
- **Persian Dataset**: [https://huggingface.co/datasets/mshojaei77/persian-gk](https://huggingface.co/datasets/mshojaei77/persian-gk)

---

*This notebook provides a complete pipeline for fine-tuning Gemma 3N for Persian language tasks using memory-efficient QLoRA techniques.*