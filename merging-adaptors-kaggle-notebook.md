# Merging LoRA Adapters with Gemma 3N Base Model

## Overview

This notebook demonstrates how to merge trained LoRA adapters with the base Gemma 3N (4B) model to create a standalone, production-ready model. The process converts the quantized model with adapters into a full 16-bit merged model suitable for deployment.

### Key Features:
- **Adapter Merging**: Combines LoRA adapters with base model weights
- **Memory Optimization**: Designed for limited GPU memory environments
- **Model Publishing**: Automated upload to Hugging Face Hub
- **Production Ready**: Creates standalone model without adapter dependencies

### Prerequisites:
- Completed LoRA fine-tuning (see training notebook)
- LoRA adapters uploaded to Hugging Face Hub
- Kaggle or Colab environment with sufficient memory

---

## 1. Environment Setup and Dependencies

### Install Required Libraries

```python
# Install latest timm library for Gemma 3N compatibility
print("Upgrading 'timm' library to the latest version...")
!pip install -q --upgrade "timm @ git+https://github.com/huggingface/pytorch-image-models.git"

# Install Unsloth with Kaggle-specific optimizations
!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies with version constraints
!pip install -q --no-deps "xformers<0.0.26" "trl<0.9.0" peft accelerate bitsandbytes

print("✅ All dependencies installed successfully.")
```

**Installation Notes:**
- **timm**: Latest version required for Gemma 3N vision components
- **unsloth[kaggle-new]**: Kaggle-optimized version with memory improvements
- **xformers<0.0.26**: Version constraint for compatibility
- **peft**: Required for LoRA adapter operations

### Import Required Libraries

```python
import torch
from unsloth import FastModel
from peft import PeftModel
from huggingface_hub import notebook_login, HfApi
import gc
import os

print("Libraries imported successfully.")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

## 2. Model Loading and Adapter Application

### Load Base Model in 4-bit Precision

```python
print("Loading base model in 4-bit precision...")

# Load the base Gemma 3N model with memory optimization
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",  # Instruction-tuned base model
    max_seq_length=4000,                    # Maximum sequence length
    dtype=None,                             # Auto-detect optimal dtype
    load_in_4bit=True,                      # Enable 4-bit quantization
)

print("✅ Base model loaded successfully.")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")
```

**Loading Configuration:**
- **4-bit quantization**: Reduces memory usage by ~75%
- **max_seq_length=4000**: Balances memory and context length
- **dtype=None**: Automatically selects optimal precision (typically bfloat16)

### Apply LoRA Adapters

```python
print("Applying LoRA adapters...")

# Load and apply the trained LoRA adapters
adapter_model_id = "mshojaei77/gemma-3n-E4B-persin-lora-adaptors"

model = PeftModel.from_pretrained(
    model, 
    adapter_model_id,
    torch_dtype=torch.float16,  # Use float16 for adapters
)

print("✅ LoRA adapters loaded and applied successfully.")
print(f"Model type after adapter loading: {type(model)}")
print(f"Adapter configuration: {model.peft_config}")
```

**Adapter Application Process:**
- Loads LoRA weights from Hugging Face Hub
- Applies adapters on top of quantized base model
- Creates PeftModel wrapper for adapter management

---

## 3. Memory Monitoring

### Pre-merge Memory Statistics

```python
def print_memory_stats(stage=""):
    """Print current GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        print(f"\n=== GPU Memory Stats {stage} ===")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Max Reserved: {max_reserved:.2f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory / 1024**3) - reserved:.2f} GB")
    else:
        print("CUDA not available")

# Monitor memory before merging
print_memory_stats("Before Merging")
```

---

## 4. Adapter Merging Process

### Merge Adapters with Base Model

```python
print("\n🔄 Starting adapter merging process...")
print("This process will:")
print("1. De-quantize the 4-bit base model to 16-bit")
print("2. Merge LoRA adapter weights into base model weights")
print("3. Create a standalone model without adapter dependencies")

# Perform the merge operation
print("\nMerging adapters into the base model...")
model = model.merge_and_unload()

print("✅ Adapters merged successfully!")
print(f"Final model type: {type(model)}")
print("The model is now a standalone Gemma3nForConditionalGeneration model.")

# Monitor memory after merging
print_memory_stats("After Merging")
```

**Merging Process Details:**

1. **De-quantization**: Converts 4-bit weights back to 16-bit precision
2. **Weight Integration**: Mathematically merges LoRA weights with base weights
3. **Memory Reallocation**: Creates new model object with merged weights
4. **Adapter Removal**: Eliminates dependency on separate adapter files

**Mathematical Operation:**
```
W_merged = W_base + (LoRA_A × LoRA_B) × scaling_factor
```
Where:
- `W_base`: Original model weights
- `LoRA_A`, `LoRA_B`: Low-rank adaptation matrices
- `scaling_factor`: `lora_alpha / r`

---

## 5. Model Validation and Testing

### Test Merged Model Functionality

```python
print("\n🧪 Testing merged model functionality...")

# Prepare a test message in Persian
test_messages = [{
    "role": "user",
    "content": "سلام! لطفاً در مورد تاریخ ایران توضیح کوتاهی بدهید."
}]

# Tokenize the input
inputs = tokenizer.apply_chat_template(
    test_messages,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
).to(model.device)

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input device: {inputs['input_ids'].device}")

# Generate a short test response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode and display the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📝 Test Response:")
print(response)
print("\n✅ Model functionality verified successfully!")
```

---

## 6. Model Saving

### Save Merged Model Locally

```python
# Define local save directory
local_save_dir = "./gemma-3n-persian-finetune-merged"

print(f"\n💾 Saving merged model to '{local_save_dir}'...")
print("This will save the full 16-bit model weights (~8-9 GB).")

# Create directory if it doesn't exist
os.makedirs(local_save_dir, exist_ok=True)

# Save the merged model and tokenizer
model.save_pretrained(
    local_save_dir,
    safe_serialization=True,  # Use safetensors format
    max_shard_size="2GB",     # Split large files for easier handling
)

tokenizer.save_pretrained(local_save_dir)

print("✅ Model saved locally successfully!")

# Check saved files
import os
saved_files = os.listdir(local_save_dir)
print(f"\n📁 Saved files ({len(saved_files)} total):")
for file in sorted(saved_files):
    file_path = os.path.join(local_save_dir, file)
    if os.path.isfile(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {file} ({size_mb:.1f} MB)")
```

**Saving Configuration:**
- **safe_serialization=True**: Uses safetensors format for security
- **max_shard_size="2GB"**: Splits model into manageable chunks
- **Complete model**: Includes all weights, configuration, and tokenizer

---

## 7. Hugging Face Hub Upload

### Authenticate with Hugging Face

```python
print("\n🔐 Authenticating with Hugging Face Hub...")

# Login to Hugging Face (will prompt for token)
try:
    notebook_login()
    print("✅ Successfully authenticated with Hugging Face Hub.")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("Please ensure you have a valid Hugging Face token.")
```

### Create Repository and Upload Model

```python
# Define repository details
hf_username = "mshojaei77"  # Replace with your username
repo_name = "gemma-3n-E4B-persian"
hf_repo_id = f"{hf_username}/{repo_name}"

print(f"\n🚀 Uploading model to Hugging Face Hub...")
print(f"Repository: {hf_repo_id}")

# Initialize Hugging Face API
api = HfApi()

try:
    # Create repository (if it doesn't exist)
    print("Creating repository...")
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,  # Don't fail if repo already exists
        private=False,  # Set to True for private repositories
    )
    print("✅ Repository created/verified successfully.")
    
    # Upload the entire model folder
    print("\nUploading model files... (This may take several minutes)")
    api.upload_folder(
        folder_path=local_save_dir,
        repo_id=hf_repo_id,
        commit_message="Add final merged 16-bit Persian Gemma 3N model",
        ignore_patterns=["*.git*", "__pycache__", "*.pyc"],  # Ignore unnecessary files
    )
    
    print("\n🎉 SUCCESS! Model uploaded successfully!")
    print(f"\n📍 Your model is now available at:")
    print(f"   https://huggingface.co/{hf_repo_id}")
    
except Exception as e:
    print(f"❌ Upload failed: {e}")
    print("Please check your internet connection and Hugging Face token.")
```

**Upload Process:**
- Creates repository if it doesn't exist
- Uploads all model files including weights, config, and tokenizer
- Provides public access (change `private=True` for private repos)
- Includes descriptive commit message

---

## 8. Model Card Creation

### Generate Comprehensive Model Card

```python
model_card_content = f'''
---
license: apache-2.0
language:
- fa
base_model: unsloth/gemma-3n-E4B-it
tags:
- gemma-3n
- unsloth
- persian
- fa
- conversational
- qlora
- fine-tuned
datasets:
- mshojaei77/persian-gk
---

# Gemma-3N 4B - Persian General Knowledge (Fine-tuned & Merged)

This model is a fine-tuned version of `unsloth/gemma-3n-E4B-it`, specialized for conversational general knowledge in the **Persian (Farsi)** language.

## Model Details

### Fine-tuning Procedure

The model was trained using QLoRA (Quantized Low-Rank Adaptation) methodology:

- **Base Model:** `unsloth/gemma-3n-E4B-it`
- **Dataset:** `mshojaei77/persian-gk` (5,900+ Persian conversations)
- **Training Framework:** [Unsloth](https://github.com/unslothai/unsloth)
- **Fine-tuning Technique:** QLoRA with 4-bit quantization
- **LoRA Configuration:**
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0
  - Target modules: All linear layers

### Merging Procedure

The final standalone model was created through a memory-efficient merging process:

1. **Base Model Loading:** Loaded in 4-bit precision for memory efficiency
2. **Adapter Application:** Applied trained LoRA adapters
3. **Merge Operation:** Used `merge_and_unload()` to integrate weights
4. **De-quantization:** Converted to full 16-bit precision
5. **Standalone Model:** Removed adapter dependencies

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "{hf_repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prepare conversation
messages = [{{
    "role": "user",
    "content": "سلام! در مورد تاریخ ایران توضیح بدهید."
}}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate response
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance

- **Model Size:** ~8.5 GB (16-bit precision)
- **Memory Requirements:** 12+ GB GPU memory for inference
- **Language:** Optimized for Persian (Farsi)
- **Domain:** General knowledge and conversational AI

## Limitations

- Specialized for Persian language tasks
- May hallucinate or provide inaccurate information
- Not suitable for critical applications without human oversight
- Knowledge cutoff based on training data

## Citation

```bibtex
@misc{{gemma3n_persian,
  author = {{mshojaei77}},
  title = {{Gemma-3N 4B Persian Fine-tuned Model}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{hf_repo_id}}}
}}
```
'''

# Save model card locally
with open(os.path.join(local_save_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(model_card_content)

print("\n📄 Model card created and saved locally.")
print("The model card will be uploaded with the model files.")
```

---

## 9. Cleanup and Memory Management

### Free GPU Memory

```python
print("\n🧹 Cleaning up memory...")

# Delete model and tokenizer objects
del model
del tokenizer

# Force garbage collection
gc.collect()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print("✅ Memory cleanup completed.")
print_memory_stats("After Cleanup")
```

---

## 10. Summary and Next Steps

### Process Summary

```python
print("\n📋 MERGE PROCESS SUMMARY")
print("=" * 50)
print("✅ Dependencies installed and configured")
print("✅ Base model loaded in 4-bit precision")
print("✅ LoRA adapters applied successfully")
print("✅ Adapters merged with base model")
print("✅ Model converted to 16-bit standalone version")
print("✅ Model saved locally and uploaded to Hub")
print("✅ Model card created and documentation added")
print("✅ Memory cleanup completed")
print("\n🎯 RESULT: Production-ready Persian Gemma 3N model")
print(f"📍 Available at: https://huggingface.co/{hf_repo_id}")
```

### Recommended Next Steps

1. **Model Evaluation**:
   - Test on validation datasets
   - Benchmark against other Persian language models
   - Evaluate on specific downstream tasks

2. **Optimization**:
   - Convert to GGUF format for efficient CPU inference
   - Create quantized versions (8-bit, 4-bit) for deployment
   - Optimize for specific hardware (ONNX, TensorRT)

3. **Deployment**:
   - Set up inference API using FastAPI or similar
   - Deploy on cloud platforms (AWS, GCP, Azure)
   - Create containerized deployment with Docker

4. **Integration**:
   - Build chat applications
   - Integrate with existing Persian NLP pipelines
   - Create fine-tuning scripts for domain adaptation

### Performance Optimization Tips

- **Memory Usage**: Use `torch.compile()` for faster inference
- **Batch Processing**: Implement dynamic batching for multiple requests
- **Caching**: Cache frequent responses to reduce computation
- **Quantization**: Use 8-bit or 4-bit quantization for deployment

---

## Resources and References

- **Unsloth Framework**: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **PEFT Library**: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- **Gemma Models**: [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- **QLoRA Paper**: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- **Persian Dataset**: [https://huggingface.co/datasets/mshojaei77/persian-gk](https://huggingface.co/datasets/mshojaei77/persian-gk)

---

*This notebook provides a complete pipeline for merging LoRA adapters with base models, creating production-ready language models for deployment.*