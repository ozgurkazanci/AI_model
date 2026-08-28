#!/bin/bash
# ASIC-AI: One-command cloud deployment for SFT training
# Usage: bash scripts/cloud/deploy_and_train.sh
#
# This script:
# 1. Clones the repo on the GPU instance
# 2. Installs dependencies
# 3. Runs SFT training with the target model
# 4. Pushes results back to GitHub
#
# Prerequisites:
# - GPU instance with CUDA (A100/H100 recommended)
# - Git configured with push access
# - At least 40GB VRAM for 35B model (or 16GB for 7B)

set -e

# ============================================================
# Configuration
# ============================================================
REPO="https://github.com/ozgurkazanci/AI_model.git"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # Override with: MODEL=... bash deploy.sh
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-0.0001}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
OUTPUT_DIR="outputs/cloud_sft"

echo "============================================================"
echo "   ASIC-AI Cloud SFT Training"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  Epochs:     $EPOCHS"
echo "  Batch:      $BATCH_SIZE x $GRAD_ACCUM"
echo "  LR:         $LR"
echo "  LoRA:       r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Max seq:    $MAX_SEQ_LEN"
echo "============================================================"

# ============================================================
# Step 1: Setup
# ============================================================
echo ""
echo "[1/5] Setting up environment..."

# Clone if needed
if [ ! -d "AI_model" ]; then
    git clone "$REPO"
fi
cd AI_model

# Install dependencies
pip install -q -r requirements.txt
pip install -q -r requirements-train.txt
pip install -q flash-attn --no-build-isolation 2>/dev/null || true

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ============================================================
# Step 2: Validate data
# ============================================================
echo ""
echo "[2/5] Validating training data..."
PYTHONPATH=src python scripts/validate_sft_data.py \
    --input data/sft/train_final.jsonl \
    --report eval_results/cloud_validation.json

# ============================================================
# Step 3: Train
# ============================================================
echo ""
echo "[3/5] Starting SFT training..."

PYTHONPATH=src python -c "
import sys, json, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Import project modules
sys.path.insert(0, 'src')
from asic_ai.data.format import SYSTEM_PROMPT

model_name = '$MODEL'
output_dir = '$OUTPUT_DIR'
epochs = $EPOCHS
batch_size = $BATCH_SIZE
grad_accum = $GRAD_ACCUM
lr = $LR
lora_r = $LORA_R
lora_alpha = $LORA_ALPHA
max_seq_len = $MAX_SEQ_LEN

print(f'Loading model: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='auto',
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA
lora_config = LoraConfig(
    r=lora_r, lora_alpha=lora_alpha,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    task_type='CAUSAL_LM', bias='none',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load data
print('Loading training data...')
examples = []
with open('data/sft/train_final.jsonl', encoding='utf-8') as f:
    for line in f:
        ex = json.loads(line.strip())
        if 'messages' in ex:
            examples.append(ex)
print(f'Loaded {len(examples)} examples')

# Format for training
def format_example(ex):
    text = tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, truncation=True, max_length=max_seq_len, padding='max_length')
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

ds = Dataset.from_list(examples)
ds = ds.map(format_example, remove_columns=ds.column_names)
print(f'Tokenized: {len(ds)} examples')

# Training
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=lr,
    warmup_steps=20,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    report_to='none',
    max_grad_norm=1.0,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
)

trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)

t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f'Training complete in {train_time/60:.1f} minutes')

# Save final
final_dir = f'{output_dir}/final'
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

# Save info
info = {
    'base_model': model_name,
    'lora_r': lora_r, 'lora_alpha': lora_alpha,
    'epochs': epochs, 'examples': len(examples),
    'train_time_min': round(train_time/60, 1),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
}
Path(f'{final_dir}/training_info.json').write_text(json.dumps(info, indent=2))
print(f'Model saved to {final_dir}')
"

# ============================================================
# Step 4: Evaluate
# ============================================================
echo ""
echo "[4/5] Running quick evaluation..."
PYTHONPATH=src python scripts/run_eval.py --model "$OUTPUT_DIR/final" --limit 5 || true

# ============================================================
# Step 5: Push results
# ============================================================
echo ""
echo "[5/5] Pushing results to GitHub..."
git add -A
git commit -m "Cloud SFT training: $MODEL, $EPOCHS epochs, $(date -Iseconds)" || true
git push origin main || true

echo ""
echo "============================================================"
echo "   Training Complete!"
echo "============================================================"
echo "  Model: $OUTPUT_DIR/final"
echo "  To merge LoRA: PYTHONPATH=src python scripts/merge_lora.py --model $OUTPUT_DIR/final"
echo "  To export GGUF: PYTHONPATH=src python scripts/export_gguf.py --model $OUTPUT_DIR/final"
echo "============================================================"
