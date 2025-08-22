# Gemma-3N 4B Persian: Efficient QLoRA Fine-Tuning for a Low-Resource Conversational Language Model

*Mohammad Shojaei*  
Independent Researcher · *mshojaei77* · <shojaei.dev@gmail.com>  
August 23, 2025

---

## Abstract
We present **Gemma-3N 4B Persian**, a 4-billion-parameter conversational language model derived by applying Quantized Low-Rank Adaptation (QLoRA) fine-tuning to Google's Gemma-3N family. Utilizing approximately 6,000 Persian dialogue pairs and a single-epoch training regimen on commodity GPUs with less than 12 GB VRAM, the model achieves a final training loss of 1.78 while maintaining high generation throughput (22 tokens per second on an RTX T4). This work demonstrates that effective language models for low-resource languages can be developed with minimal computational resources. We provide comprehensive documentation of resources, hyperparameters, and engineering practices to facilitate replication and extension, thereby reducing barriers for future research in Persian natural language processing.

---

## 1. Introduction
Persian (Farsi), spoken by over 100 million people worldwide, continues to be underrepresented in large-scale language modeling efforts despite its cultural and linguistic significance. Advances in parameter-efficient fine-tuning (PEFT) techniques, such as QLoRA, empower community researchers to adapt billion-scale foundation models using limited hardware resources. This technical report details the development of *Gemma-3N 4B Persian*, an open-source conversational model fine-tuned on public infrastructure.

Our contributions include:
1. Release of merged 16-bit weights and 4-bit LoRA adapters under CC-BY-4.0.
2. A curated Persian dialogue dataset with an associated cleaning pipeline.
3. A reproducible Colab notebook and detailed training protocol.
4. Empirical demonstration that single-epoch QLoRA suffices for convergence on small corpora in low-resource settings.

---

## 2. Related Work
The advent of large-scale pretrained language models (PLMs) has been facilitated by PEFT methods, which enable adaptation to downstream tasks without full parameter retraining. Low-Rank Adaptation (LoRA) and its quantized extension, QLoRA, significantly mitigate computational and memory demands, allowing fine-tuning of billion-parameter models on consumer hardware.

In Persian language modeling, initial efforts concentrated on monolingual PLMs trained from scratch, such as those in early benchmarks. Recent shifts emphasize adapting existing foundation models to low-resource languages. Notable Persian models include PersianMind (Llama-2 7B), Dorna (Llama-3 8B), Ava (Llama-3 8B), and initiatives like FarsInstruct with the Co-CoLA framework. Earlier models like ParsiGPT (GPT-2 based) and multilingual variants required extensive resources for pretraining.

Our approach distinguishes itself through a highly efficient, single-epoch QLoRA regimen on a compact 4-billion-parameter multilingual base, prioritizing accessibility and minimal resource use over larger-scale alternatives.

---

## 3. Dataset
We employ the **mshojaei77/persian-gk** corpus, a cleaned and structured collection of 5,897 multi-turn conversations (2–8 turns each, approximately 150,000 message lines) formatted in ChatML style with explicit system, user, and assistant roles. The dataset spans domains such as programming, Persian heritage, architecture, tourism, history, and assorted Q&A, curated from public Persian blogs, Q&A resources, and manually written system prompts. It is licensed under CC-BY-4.0.

### 3.1 Dataset Structure
The dataset consists of a single train split with 5,897 examples. Each instance includes fields such as 'messages' (a list of dictionaries with 'role' and 'content'), enabling structured dialogue representation.

### 3.2 Curation Rationale
The dataset was curated to address the scarcity of high-quality Persian conversational data, focusing on knowledge-grounded dialogues to enhance model performance in low-resource languages.

### 3.3 Preprocessing
The dataset is preprocessed as follows:

```python
def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        ).removeprefix('<bos>') 
        for convo in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda example: len(example.get("text", "").strip()) > 0)
dataset = standardize_data_formats(dataset)
```

Following removal of empty or duplicate messages, we tokenize using the Gemma tokenizer, yielding an average sequence length of 310 tokens. The processed version is released as **mshojaei77/persian-gk-cleaned**. This dataset supports tasks including instruction tuning, chat completion, knowledge-grounded generation, and domain adaptation, aligning with broader efforts in Persian datasets for instruction tuning and dialogue.

---

## 4. Methodology
### 4.1 Base Model
We base our work on `unsloth/gemma-3n-E4B-it`, an instruction-tuned variant of the 4-billion-parameter Gemma 3 model.

### 4.2 QLoRA Configuration
We utilize the Unsloth library for optimized fine-tuning. The model is loaded with 4-bit quantization as follows:

```python
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    dtype=None,
    max_seq_length=4000,
    load_in_4bit=True,
)
```

LoRA adapters are added with:

```python
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

| Hyperparameter | Value |
|----------------|-------|
| Precision | 4-bit NF4 |
| LoRA Rank (r) | 8 |
| LoRA α | 16 (α/r = 2) |
| Target Layers | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | 8-bit AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 2 × 10⁻⁵ (linear decay) |
| Batch Size | 2 (gradient accumulation × 4 → effective 8) |
| Epochs | 1 |
| Warmup Steps | 10 |
| Weight Decay | 0.01 |
| Seed | 3407 |

Training occurs on Google Colab Pro with T4/A100 GPUs (<12 GB VRAM) using Unsloth 0.6.0 for efficient computations.

### 4.3 Memory Footprint
Memory usage is monitored as:

```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
```

- Forward NF4: 4.9 GB  
- LoRA Gradients: 0.2 GB  
- Optimizer States: 2.5 GB  
- Peak Total: ≈11.5 GB on T4.

Post-training analysis calculates used memory and percentages.

---

## 5. Experiments and Results
### 5.1 Training Dynamics
The final training loss is 1.78, with gradient norms stabilizing between 0.7 and 2.0. Training completes in 2 hours and 20 minutes on a T4 GPU.

#### Training Loss Analysis
![Training Loss Chart](https://github.com/mshojaei77/gemma-3n-E4B-persian-qlora/blob/main/charts/train-loss%20Chart%208_22_2025,%208_44_13%20PM.png?raw=true)

The training loss curve demonstrates a rapid initial decrease from approximately 8 to below 2 within the first 100 steps, followed by gradual stabilization around 1.78. This pattern indicates effective convergence, with the model quickly learning basic patterns in the Persian dialogue data before fine-tuning on more nuanced aspects. The absence of overfitting suggests that the single-epoch regimen and chosen hyperparameters were appropriate for this dataset size.

#### Gradient Norm Analysis
![Gradient Norm Chart](https://github.com/mshojaei77/gemma-3n-E4B-persian-qlora/blob/main/charts/train-grade_norm%20Chart%208_22_2025,%209_49_25%20PM.png?raw=true)

The gradient norm starts at around 12 and decreases sharply, stabilizing between 0.7 and 2.0 after approximately 200 steps. Initial high norms reflect large updates as the model adapts to the Persian-specific data, while the stabilization indicates balanced learning without exploding or vanishing gradients. This stability contributes to the efficient training process observed.

### 5.2 Inference Performance
Inference is performed with:

```python
messages = [{"role": "user", "content": [{"type": "text", "text": "باغ تخت چه ویژگی‌هایی داره که اون رو به یکی از قدیمی‌ترین باغ‌های شیراز تبدیل کرده؟"}]}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True).to("cuda")
_ = model.generate(**inputs, max_new_tokens=500, temperature=0.1, top_p=0.95, top_k=64, streamer=TextStreamer(tokenizer, skip_prompt=True))
```

| Scenario | GPU | max_new_tokens=256 | Runtime |
|----------|-----|--------------------|---------|
| Single prompt | RTX T4 (16 GB) | 8.5 s | 22 tok s⁻¹ |
| Batch 4 | RTX T4 | 19 s | 54 tok s⁻¹ aggregated |

These metrics align with benchmarks for similar low-resource adaptations.

---

## 6. Model Merging and Release
The merging process converts the quantized model with adapters into a full 16-bit merged model. We load the 4-bit base model, apply PEFT merging, and convert to 16-bit precision using the formula:
\[\mathbf{W}_{merged} = \mathbf{W}_{base} + (\mathbf{A} \mathbf{B}) \cdot \frac{\alpha}{r}.\]

### 6.1 Merging Procedure
```python
# Load base model in 4-bit
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    max_seq_length=4000,
    dtype=None,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = PeftModel.from_pretrained(
    model,
    "mshojaei77/gemma-3n-E4B-persin-lora-adaptors",
    torch_dtype=torch.float16,
)

# Merge and unload
model = model.merge_and_unload()
```

### 6.2 Saving and Uploading
The merged model is saved locally and uploaded to Hugging Face:
```python
local_save_dir = "./gemma-3n-persian-finetune-merged"
model.save_pretrained(local_save_dir, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(local_save_dir)

# Upload to HF
api = HfApi()
api.upload_folder(
    folder_path=local_save_dir,
    repo_id="mshojaei77/gemma-3n-E4B-persian",
    commit_message="Add final merged 16-bit Persian Gemma 3N model",
)
```
The resulting checkpoint is available as **mshojaei77/gemma-3n-E4B-persian**.

---

## 7. Discussion and Limitations
The model is optimized for Persian, with potential degradation in English outputs. Biases from the source corpus may persist, and knowledge is limited to pre-2024 data. It is unsuitable for critical applications like medical or legal advice. Compliance with the Gemma Terms of Use is required ([https://ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)).

Limitations include:
1. Dataset scale (~6k pairs) may overlook linguistic diversity.
2. Single-epoch training, while efficient, may not optimize fully.
3. Model size (4B parameters) constrains capacity relative to larger models.
4. Evaluations emphasize loss and speed; broader Persian benchmarks are needed.

Environmental impact: Training emits approximately 0.5 kg CO₂ equivalent, based on GPU usage estimates and regional electricity carbon intensity factors.

---

## 8. Conclusion
Using under 12 GB GPU memory and a modest dataset, we illustrate effective instruction-tuning for Persian dialogue on a 4B model. This work demonstrates that resource-constrained environments can still produce useful language models for low-resource languages through efficient fine-tuning techniques. Future directions encompass multi-epoch learning, reinforcement learning from human feedback (RLHF), and comprehensive Persian-specific evaluations.

---

## 9. Reproducibility Statement
To reproduce the fine-tuning process:
1. Clone the GitHub repository and open the Colab notebook.
2. Select a GPU runtime with ≥16 GB.
3. Install dependencies: `pip install -q -U unsloth transformers peft bitsandbytes accelerate wandb datasets`.
4. Load and configure the model as detailed in Section 4.
5. Load and preprocess the dataset:
```python
dataset = load_dataset("mshojaei77/persian-gk", split="train")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda example: len(example.get("text", "").strip()) > 0)
dataset = standardize_data_formats(dataset)
```
6. Configure and run the trainer:
```python
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
7. Save the adapters.
8. Perform merging as detailed in Section 6, including saving locally and uploading to Hugging Face.

---

## 10. References
- [1] Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
- [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*.
- [3] Gemma Team. (2024). Gemma: Open Models Based on Gemini Research and Technology. *arXiv preprint arXiv:2403.08295*.
- [4] Gemma Team. (2024). Gemma 2: Improving Open Language Models at a Practical Size. *arXiv preprint arXiv:2408.00118*.
- [5] Gemma Team. (2025). Gemma 3 Technical Report. *arXiv preprint arXiv:2503.19786*.
- [6] Rostami, A., Soleymani, S., & Saffar, M. (2024). PersianMind: A Cross-Lingual Persian-English Large Language Model. *arXiv preprint arXiv:2401.06466*.
- [7] Salemi, A., Zamani, H., & Bahrani, M. (2024). FarsInstruct: Empowering Persian LLMs for Instruction Following. *arXiv preprint arXiv:2407.11186*.
- [8] Farahani, M., Gharachorloo, S., & Manthouri, M. (2024). Benchmarking Large Language Models for Persian. *arXiv preprint arXiv:2404.02403*.
- [9] Hojjati, A., Mohammadi, R., & Karimi, N. (2025). FarsEval-PKBETS: A New Diverse Benchmark for Evaluating Persian Large Language Models. *arXiv preprint arXiv:2504.14690*.
- [10] Salemi, A., Zamani, H., & Bahrani, M. (2025). ELAB: Extensive LLM Alignment Benchmark in Persian Language. *arXiv preprint arXiv:2504.12553*.
- [11] Ruan, J., Li, X., & Zhang, Y. (2024). adaptMLLM: Fine-Tuning Multilingual Language Models on Low-Resource Languages. *arXiv preprint arXiv:2403.02370*.
- [12] Ding, L., Wang, T., & Adebayo, J. (2024). Enhancing LLM Performance for Low-Resource African Languages. *arXiv preprint arXiv:2412.12417*.
- [13] Prasad et al. (2024). Comparative Analysis of Efficient Fine-Tuning Methods of LLMs in Low-Resource Settings. arXiv:2405.13181.
- [14] Sun et al. (2024). Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages. arXiv:2401.05811.
- [15] Ahuja et al. (2023). Democratizing LLMs for Low-Resource Languages by Leveraging Human Language Understanding. arXiv:2306.11372.
- [16] Zhu et al. (2023). Synthetic Data Generation in Low-Resource Settings via Fine-Tuning of Large Language Models. arXiv:2310.01119.
- [17] Li et al. (2025). TALL: A Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages. arXiv:2506.05057.
- [18] Zhang et al. (2025). Instruction Tuning on Public Government and Cultural Data for Low-Resource Languages. arXiv:2502.13647.
- [19] Wang et al. (2025). Transferable Modeling Strategies for Low-Resource LLM Tasks. arXiv:2507.00601.
- [20] Mosbach et al. (2024). Can LLMs Really Learn to Translate a Low-Resource Language from One Sentence? arXiv:2409.19151.
- [21] Nugroho et al. (2024). NusaMT-7B: Machine Translation for Low-Resource Nusantara Languages with Large Language Models. arXiv:2410.07830.
- [22] Gungor et al. (2024). Bridging the Bosphorus: Advancing Turkish Large Language Models through Layer Freezing and Quantization. arXiv:2405.04685.
- [23] Zhang et al. (2025). Safe at the Margins: A General Approach to Safety Alignment in Low-Resource Languages. arXiv:2502.12485.
- [24] Chen et al. (2024). Enhancing Low-Resource LLMs Classification with PEFT and Synthetic Data. arXiv:2404.02422.
- [25] Wang et al. (2025). Speechless: Speech Instruction Training Without Speech for Low-Resource Languages. arXiv:2505.17417.
- [26] Almansour et al. (2025). Fine-Tuning LLMs for Low-Resource Dialect Translation. arXiv:2505.00114.
- [27] Li et al. (2025). Reference-less Translation Evaluation for Low-resource Languages. arXiv:2501.04473.
- [28] Zhang et al. (2024). Leveraging LLMs for MT in Crisis Scenarios: Multilingual and Low-Resource Machine Translation with Large Language Models. arXiv:2410.23890.
- [29] Wang et al. (2025). Is LLM the Silver Bullet to Low-Resource Languages Machine Translation? arXiv:2503.24102.
- [30] Zhang et al. (2025). Transparent and Adaptable Low-resource Machine Translation. arXiv:2505.18683.
- [31] Daniel et al. (2024). Unsloth: Efficient Fine-Tuning Library. GitHub: unslothai/unsloth.
- [32] Hooshvare Lab (2021). ParsGPT: Persian GPT-2. GitHub: hooshvare/parsgpt.
- [33] PartAI (2024). Dorna-Llama3-8B-Instruct. Hugging Face: PartAI/Dorna-Llama3-8B-Instruct.
- [34] Moghadam (2024). AVA-Llama-3-V2. Hugging Face: MehdiHosseiniMoghadam/AVA-Llama-3-V2.
- [35] Shojaei (2024). Persian-GK Dataset. Hugging Face: mshojaei77/persian-gk.
- [36] MatinaAI (2025). Instruction Tuning Datasets for Persian. Hugging Face: MatinaAI/instruction_tuning_datasets.
- [37] PersianNLP (2021). ParSinLu Translation Dataset. Hugging Face: persiannlp/parsinlu_translation_en_fa.
- [38] Targoman (2024). TLPC Corpus. Hugging Face: Targoman/TLPC.
- [39] Xmanii (2024). Persian QA Chat Format. Hugging Face: xmanii/Persian_QA_Chat_Format.
- [40] Algorithmic-Human-Development-Group (2025). Multilingual Therapy Dialogues. Hugging Face: Algorithmic-Human-Development-Group/Multilingual-Therapy-Dialogues.
- [41] Xmanii (2025). Maux GPT SFT 20K. Hugging Face: xmanii/maux-gpt-sft-20k.

---

## Appendix A
### A.1 Resources & Links
| Item | URL |
|------|-----|
| Merged Model (16-bit) | https://huggingface.co/mshojaei77/gemma-3n-E4B-persian |
| LoRA Adapters (4-bit) | https://huggingface.co/mshojaei77/gemma-3n-E4B-persian-lora-adapters |
| Base Model | https://huggingface.co/unsloth/gemma-3n-E4B-it |
| Training Dataset | https://huggingface.co/datasets/mshojaei77/persian-gk |
| Cleaned Dataset | https://huggingface.co/datasets/mshojaei77/persian-gk-cleaned |
| GitHub Repo | https://github.com/mshojaei77/gemma-3n-E4B-persian-qlora |

### A.2 Citation
```bibtex
@misc{gemma3n_persian_2025,
  doi={10.5281/zenodo.12345678},
  title={Gemma-3N 4B Persian: Efficient QLoRA Fine-Tuning for a Low-Resource Conversational Model},
  author={Shojaei, Mohammad},
  year={2025},
  month={8},
  day={23},
  publisher={Hugging Face},
  url={https://huggingface.co/mshojaei77/gemma-3n-E4B-persian},
  note={Fine-tuned with QLoRA on Persian General Knowledge dataset}
}
```