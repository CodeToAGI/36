"""
EP36 Challenge Solution — LoRA Fine-Tune TinyLlama-1.1B on Custom Q&A
Requires: pip install transformers peft trl bitsandbytes datasets accelerate torch
Runs on free Colab T4 (or any GPU with ≥8 GB).
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "ep36-lora-output"
ADAPTER_DIR = "ep36-lora-adapter"
MERGED_DIR = "ep36-merged-model"

# ── 1. 4-bit quantisation config ──────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # needed for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 2. LoRA config ────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # add "k_proj","o_proj" for stronger results
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected ≈ 2.4 M trainable / 1.1 B total (~0.22 %)

# ── 3. Tiny custom dataset (replace with your 200 real Q&A pairs) ─────────────
# Format must be Alpaca-style so the model learns the ### Instruction / ### Response pattern.
raw_data = [
    {
        "text": (
            "### Instruction:\nWhat is gradient descent?\n\n"
            "### Response:\nGradient descent minimises a loss function by iteratively "
            "moving in the direction of steepest descent. Update rule: θ ← θ − η∇L(θ)."
        )
    },
    {
        "text": (
            "### Instruction:\nExplain LoRA in one sentence.\n\n"
            "### Response:\nLoRA freezes the original weights W₀ and trains two small "
            "low-rank matrices A and B so that ΔW = B·A, updating only ~0.1 % of parameters."
        )
    },
    # … add 198 more high-quality pairs here …
]

# Quick synthetic expansion for demo (delete when you have real data)
for i in range(50):
    raw_data.append({
        "text": (
            f"### Instruction:\nWhat is the capital of country number {i}?\n\n"
            f"### Response:\nThis is a placeholder answer for demo purposes only."
        )
    })

dataset = Dataset.from_list(raw_data)

# ── 4. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,          # effective batch size = 8
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",               # memory-friendly
    report_to="none",
)

# ── 5. SFTTrainer ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

print("Starting training…")
trainer.train()
trainer.save_model(ADAPTER_DIR)
print(f"Adapter saved → {ADAPTER_DIR}")

# ── 6. Merge LoRA weights for deployment ──────────────────────────────────────
print("Merging LoRA weights…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
merged = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged = merged.merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved → {MERGED_DIR}")

# ── 7. Quick before/after inference demo ──────────────────────────────────────
def generate(model, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = (
    "### Instruction:\nWhat is gradient descent?\n\n"
    "### Response:\n"
)

print("\n── BEFORE (base model) ──")
# reload clean base for comparison
base_clean = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
print(generate(base_clean, test_prompt))

print("\n── AFTER (merged LoRA model) ──")
print(generate(merged, test_prompt))

print("\nDone! Log the loss curve from the training run and post a before/after screenshot.")
