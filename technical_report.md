# Technical Report – Gemma-3N 4B Persian QLoRA Project

This document records every technical detail required to **reproduce, audit or extend** the fine-tuning of Google’s Gemma-3N 4 B model for Persian conversational tasks.

---

## 1. Resources & Public Links

| Item | URL |
|------|-----|
| Final Merged Model (16-bit) | https://huggingface.co/mshojaei77/gemma-3n-E4B-persian |
| LoRA Adapters (4-bit) | https://huggingface.co/mshojaei77/gemma-3n-E4B-persian-lora-adapters |
| Base Model | https://huggingface.co/unsloth/gemma-3n-E4B-it |
| Training Dataset | https://huggingface.co/datasets/mshojaei77/persian-gk |
| Cleaned Dataset | https://huggingface.co/datasets/mshojaei77/persian-gk-cleaned |
| GitHub Repo (code & notebooks) | https://github.com/mshojaei77/gemma-3n-E4B-persin-qlora |

---

## 2. Hardware & Environment

* **Platform:** Google Colab Pro (T4 / A100 depending on session)
* **Python:** 3.10
* **CUDA:** 12.x
* **Frameworks:**
  * `transformers>=4.39`
  * `accelerate>=0.27`
  * `peft>=0.10`
  * `unsloth>=0.6.0`
  * `bitsandbytes>=0.43`
  * `trl`, `wandb`, `datasets`
* **GPU Memory Budget:** ≤ 12 GB

A full `requirements.txt` is embedded in `training_qlora-4bit-colab-notebook.md`.

---

## 3. Dataset Preparation

| Metric | Value |
|--------|-------|
| Conversations | 5 ,897 |
| Format | ChatML (`<start_of_turn>role … <end_of_turn>`) |
| Roles | `system`, `user`, `assistant` |
| Domains | Persian history & culture, programming, tourism, architecture |
| License | CC-BY-4.0 |

Steps:
1. Loaded `mshojaei77/persian-gk` with 🤗 Datasets.
2. Removed empty or duplicate messages (cleaned split = “persian-gk-cleaned”).
3. Tokenised using Gemma tokenizer – average length ≈ 310 tokens.

---

## 4. Fine-Tuning Configuration (QLoRA)

| Hyper-Param | Value | Notes |
|-------------|-------|-------|
| Base Model | `unsloth/gemma-3n-E4B-it` | 4 B instr-tuned Gemma |
| Precision | 4-bit NF4 | via BitsAndBytes |
| LoRA Rank (r) | 8 | |
| LoRA Alpha | 16 | scale = α/r = 2.0 |
| LoRA Dropout | 0.0 | |
| Target Layers | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | covers attention & MLP |
| Optimiser | AdamW 8-bit | betas (0.9, 0.999) |
| LR | 2 × 10⁻⁵ | linear decay |
| Batch | 2 (grad-accum ×4 → eff 8) |
| Epochs | 1 | single pass achieves convergence |
| Warm-up | 10 steps |
| Weight Decay | 0.01 |
| Seed | 3407 |

Implementation is demonstrated in **training_qlora-4bit-colab-notebook.md**. The notebook relies on [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient forward & backward passes.

### Memory Footprint

* **Forward NF4:** 4.9 GB
* **Gradients (LoRA only):** 0.2 GB
* **Optimizer States:** 2.5 GB (8-bit AdamW)
* **Peak Total:** ≈ 11.5 GB on 16 GB T4

---

## 5. Training Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 1.78 |
| Gradient Norm | stable in (0.7 – 2.0) |
| Wall-clock | 2 h 20 m (T4) |

Charts are stored in `/charts/` and visible on GitHub.

---

## 6. Adapter → Full Model Merge

1. Reloaded base model in 4-bit.
2. Loaded LoRA checkpoints and called `PeftModel.merge_and_unload()` (equivalent to equation below).
   $$\mathbf W_{merged}=\mathbf W_{base}+ (\mathbf A\mathbf B)\cdot \tfrac{\alpha}{r}$$
3. De-quantised to 16-bit for deployment.
4. Uploaded to HF Hub as **mshojaei77/gemma-3n-E4B-persian**.

---

## 7. Inference Benchmarks

| Scenario | GPU | max_new_tokens=256 | Runtime |
|----------|-----|--------------------|---------|
| Single-prompt | RTX T4 (16 GB) | 8.5 s | ≈ 22 tokens/s |
| Batch 4 | RTX T4 | 19 s | 54 tokens/s aggregated |

Decoding parameters: `temperature=0.7`, `top_p=0.95`, `do_sample=True`.

---

## 8. Limitations & Ethical Notes

* Persian-centric; English answers degrade.
* Knowledge cutoff ≈ Jan 2024.
* Potential cultural bias from source data.
* Not suitable for medical / legal reliance – human review recommended.

See [Gemma Terms of Use](https://ai.google.dev/gemma/terms) for licence obligations.

---

## 9. Reproduction Checklist

1. Clone GitHub repo & open Colab notebook.
2. Select GPU runtime ≥ 16 GB.
3. Run `pip install -q -U unsloth transformers peft bitsandbytes accelerate wandb datasets`.
4. Execute cells **1 → 6** in `training_qlora-4bit-colab-notebook.md`.
5. Optionally push adapters & merged model to your own HF account.

---

## 10. Citation

```bibtex
@misc{gemma3n_persian_2024,
  title={Gemma-3N 4B Persian Fine-tuned Model},
  author={Shojaei, M.},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/mshojaei77/gemma-3n-E4B-persian},
  note={Fine-tuned with QLoRA on Persian General Knowledge dataset}
}
```