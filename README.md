# EP36 — Fine-Tuning LLMs Explained: Full Fine-Tune, LoRA & QLoRA

**CodeToAGI Deep Learning Series · Episode 36 · Module 7**

Base language models only predict the next token.  
Fine-tuning teaches them to follow instructions.  
Full fine-tuning a 7B model needs 100–120 GB of GPU memory.  
**LoRA** (and **QLoRA**) makes the same result possible on a single free Colab T4.

## What You Will Learn
- Why fine-tuning is necessary (base LM vs instruction-following assistant)
- Memory math that makes full fine-tuning impractical for most teams
- LoRA intuition: freeze W₀, train low-rank matrices B·A
- QLoRA: NF4 4-bit quantisation + double quant + paged optimiser
- Instruction formats (Alpaca, ChatML, ShareGPT)
- Complete PEFT + TRL code path
- `merge_and_unload()` for zero-overhead deployment
- Decision guide: when to prompt vs LoRA vs full FT

## Quick Start (Challenge)
```bash
pip install transformers peft trl bitsandbytes datasets accelerate
python ep36_lora_finetune.py
