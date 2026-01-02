# OneRec-Think LLM Training Pipeline

End-to-end guide for training a Qwen3-8B model on OneRec-Think Beauty dataset with semantic IDs.

## Quick Start (One Command)

```bash
NUM_GPUS=8 DEEPSPEED_CONFIG=train/scripts/ds_config_zero3.json \
  bash run_llm_pipeline_beauty.sh
```

## Overview

This pipeline adapts the [semantic-ids-llm](https://github.com/marceloabk/semantic-ids-llm) methodology to the OneRec-Think Beauty dataset, implementing:

- **Semantic ID format conversion**: OneRec → semantic-ids-llm format with offset encoding
- **Leave-last-out evaluation**: Aligned with original OneRec-Think protocol
- **Two-stage training**: Vocabulary extension + full fine-tuning
- **5 types of training data**: ~1.1M conversations (A: sid→text, B: text→sid, C: sequential, D: semantic understanding, E: multi-hop reasoning)

## Data Split Protocol

Following the original OneRec-Think leave-last-out strategy:

```
For each user sequence with N items:
- Train:  items[:-2] → predict item[-2]
- Val:    items[:-1] → predict item[-1]
- Test:   items[:]   → predict next item (no ground truth)
```

This prevents data leakage between splits and enables realistic evaluation.

## Prerequisites

### Required Data Files

Place these files in the `data/` directory:

1. `Beauty.pretrain.json` - Item metadata with OneRec semantic IDs
2. `sequential_data_processed.txt` - User interaction sequences

### Environment Setup (Recommended)

The pipeline script `run_llm_pipeline_beauty.sh` creates a dedicated conda env and installs all required packages.

Key environment variables:
- `ENV_NAME` (default: `onerec-think-llm`)
- `FORCE_RECREATE_ENV=1` to rebuild the env
- `INSTALL_CUDA_TORCH=0` for CPU-only torch
- `SKIP_PIP_INSTALL=1` to skip installs (assumes deps already installed)

### Python Dependencies

```bash
pip install torch pandas pyarrow transformers tqdm wandb datasets trl bitsandbytes deepspeed accelerate huggingface-hub
```

### Hardware Requirements

- **Stage 1 (Vocab)**: ~16GB GPU memory (can run on CPU but slower)
- **Stage 2 (Full)**: ~40GB GPU memory (recommend A100 or use gradient checkpointing)
- **Evaluation**: ~16GB GPU memory

---

## Pipeline Steps

### Step 1: Data Preprocessing

Convert OneRec format to semantic-ids-llm format and create train/val/test splits.

```bash
python src/prep_onerec_data.py \
  --data-dir data \
  --output-dir data
```

**Input:**
- `data/Beauty.pretrain.json` (12,101 items)
- `data/sequential_data_processed.txt` (22,363 sequences)

**Output:**
- `data/Beauty_items_onerec.parquet` - Items with converted semantic IDs
- `data/Beauty_sequences_train.parquet` - Training sequences (items[:-2] → item[-2])
- `data/Beauty_sequences_val.parquet` - Validation sequences (items[:-1] → item[-1])
- `data/Beauty_sequences_test.parquet` - Test sequences (items[:] → next item)

**Semantic ID Conversion Example:**
```
OneRec format:
  <|sid_begin|><s_a_99><s_b_19><s_c_220><s_d_204><|sid_end|>

semantic-ids-llm format (with offset encoding):
  <|sid_start|><|sid_99|><|sid_275|><|sid_732|><|sid_972|><|sid_end|>

Offset: Level 0=[0-255], Level 1=[256-511], Level 2=[512-767], Level 3=[768-1023]
```

**Testing (optional):**
```bash
# Test on 100-item sample
python src/prep_onerec_data.py --limit 100
```

---

### Step 2: Generate Training Data

Generate ~1.1M ChatML conversations from the training split only.

```bash
python src/generate_training_data_onerec.py \
  --data-dir data \
  --output-dir data \
  --seed 42
```

**Input:**
- `data/Beauty_items_onerec.parquet`
- `data/Beauty_sequences_train.parquet` (uses only training split)

**Output:**
- `data/Beauty_conversations_train_onerec.parquet` (~1.1M samples)

**Training Data Types:**

| Type | Description | Count | Example |
|------|-------------|-------|---------|
| A | Semantic ID → Text | ~48K | "What is the title of product <\|sid_99\|>?" → "Hair Shampoo..." |
| B | Text → Semantic ID | ~73K | "Find product titled 'Hair Shampoo...'" → "<\|sid_99\|>..." |
| C | Sequential prediction | ~224K | "User bought X, Y. What's next?" → Z |
| D | Semantic understanding | ~11K | "What category is prefix <\|sid_99\|><\|sid_275\|>?" → "Hair Care" |
| E | Multi-hop reasoning | ~720K | "After buying X, what Y might they buy?" → Z |

**Testing (optional):**
```bash
# Limit Type C and E samples for faster testing
python src/generate_training_data_onerec.py \
  --limit-type-c 1000 \
  --limit-type-e 5000
```

**Validation:**
```bash
python src/validate_training_data.py \
  --data-path data/Beauty_conversations_train_onerec.parquet
```

---

### Step 3: Stage 1 Training - Vocabulary Extension

Train only the new semantic ID token embeddings (1024 tokens: `<|sid_0|>` to `<|sid_1023|>`).

```bash
# Single GPU
python src/finetune_qwen3_8b_vocab.py \
  --data_path data/Beauty_conversations_train_onerec.parquet \
  --output_dir models/qwen3_8b_vocab_extended \
  --learning_rate 1e-3 \
  --max_steps 1000 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 200 \
  --wandb_project onerec-llm-training

# Multi-GPU (DeepSpeed)
deepspeed --num_gpus 8 src/finetune_qwen3_8b_vocab.py \
  --deepspeed train/scripts/ds_config_zero3.json
```

**Key Parameters:**
- High learning rate (1e-3) for embedding initialization
- Only new token embeddings are trainable (~8M parameters)
- Rest of model is frozen
- Fast training: ~1-2 hours on A100

**Output:**
- `models/qwen3_8b_vocab_extended/final/` - Extended vocabulary checkpoint

**W&B Tracking:**
- Training loss
- Learning rate schedule
- Sample generations during training

---

### Step 4: Stage 2 Training - Full Fine-tuning

Fine-tune the entire model on all conversation types.

```bash
# Single GPU
python src/finetune_qwen3_8b_full.py \
  --data_path data/Beauty_conversations_train_onerec.parquet \
  --model_path models/qwen3_8b_vocab_extended/final \
  --output_dir models/qwen3_8b_stage2 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --wandb_project onerec-llm-training

# Multi-GPU (DeepSpeed)
deepspeed --num_gpus 8 src/finetune_qwen3_8b_full.py \
  --deepspeed train/scripts/ds_config_zero3.json
```

**Key Parameters:**
- Low learning rate (2e-5) for full model fine-tuning
- All ~8B parameters are trainable
- Uses LoRA/QLoRA for memory efficiency (optional)
- Training time: ~24-48 hours on A100 for 3 epochs

**Output:**
- `models/qwen3_8b_stage2/checkpoint-{step}/` - Intermediate checkpoints
- `models/qwen3_8b_stage2/final/` - Final trained model

**Memory Optimization (if needed):**
```bash
# Use 4-bit quantization + gradient checkpointing
python src/finetune_qwen3_8b_full.py \
  --load_in_4bit \
  --gradient_checkpointing \
  --batch_size 4 \
  --gradient_accumulation_steps 32
```

---

### Step 5: Evaluation

Evaluate the trained model using Hit@K and NDCG@K metrics with beam search.

#### Validation Evaluation (with ground truth)

```bash
python src/evaluate_llm.py \
  --model_path models/qwen3_8b_stage2/final \
  --data_path data/Beauty_sequences_val.parquet \
  --split_type val \
  --batch_size 4 \
  --num_beams 10 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --wandb_project onerec-llm-evaluation \
  --save_results results/eval_val.json
```

#### Test Set Prediction (no ground truth)

```bash
python src/evaluate_llm.py \
  --model_path models/qwen3_8b_stage2/final \
  --data_path data/Beauty_sequences_test.parquet \
  --split_type test \
  --batch_size 4 \
  --num_beams 10 \
  --save_results results/predictions_test.json
```

**Evaluation Metrics:**
- **Hit@K**: Fraction of cases where ground truth appears in top-K predictions
- **NDCG@K**: Normalized Discounted Cumulative Gain at K (position-aware)

**W&B Tracking:**
- Interim metrics every 5% of batches (20 checkpoints total)
- Final metrics: Hit@1/5/10, NDCG@5/10
- Metrics comparison table
- Sample predictions

**Verbose Mode (for debugging):**
```bash
python src/evaluate_llm.py \
  --model_path models/qwen3_8b_stage2/final \
  --data_path data/Beauty_sequences_val.parquet \
  --split_type val \
  --verbose
```

---

## Project Structure

```
OneRec-Think/
├── data/
│   ├── Beauty.pretrain.json              # Input: Item metadata
│   ├── sequential_data_processed.txt     # Input: User sequences
│   ├── Beauty_items_onerec.parquet       # Output: Converted items
│   ├── Beauty_sequences_train.parquet    # Output: Training split
│   ├── Beauty_sequences_val.parquet      # Output: Validation split
│   ├── Beauty_sequences_test.parquet     # Output: Test split
│   └── Beauty_conversations_train_onerec.parquet  # Output: Training data
├── src/
│   ├── prep_onerec_data.py               # Step 1: Data preprocessing
│   ├── generate_training_data_onerec.py  # Step 2: Training data generation
│   ├── finetune_qwen3_8b_vocab.py        # Step 3: Stage 1 training
│   ├── finetune_qwen3_8b_full.py         # Step 4: Stage 2 training
│   ├── evaluate_llm.py                   # Step 5: Evaluation
│   ├── validate_training_data.py         # Utilities: Data validation
│   └── utils.py                          # Utilities: Logger, device manager
├── models/
│   ├── qwen3_8b_vocab_extended/          # Stage 1 output
│   └── qwen3_8b_stage2/                  # Stage 2 output
├── logs/                                 # Training and evaluation logs
├── results/                              # Evaluation results
└── README_LLM_TRAINING.md                # This file
```

---

## Quick Start (Full Pipeline)

Run all steps sequentially:

```bash
# End-to-end pipeline (creates conda env, installs deps, runs all steps)
NUM_GPUS=8 DEEPSPEED_CONFIG=train/scripts/ds_config_zero3.json \
  bash run_llm_pipeline_beauty.sh
```

Optional environment variables for `run_llm_pipeline_beauty.sh`:
- `DATA_DIR`, `RESULTS_DIR`, `LOG_DIR`
- `HOSTFILE` (for multi-node DeepSpeed)
- `DISABLE_WANDB=1` to disable W&B logging

---

## Expected Results

Based on semantic-ids-llm paper and OneRec-Think benchmarks:

| Metric | Expected Range |
|--------|---------------|
| Hit@1  | 5-15% |
| Hit@5  | 15-30% |
| Hit@10 | 25-45% |
| NDCG@5 | 0.15-0.30 |
| NDCG@10| 0.20-0.35 |

*Note: Results depend on hyperparameters, training epochs, and data quality.*

---

## Troubleshooting

### Out of Memory (OOM)

**Stage 1:**
```bash
# Reduce batch size
python src/finetune_qwen3_8b_vocab.py --batch_size 8 --gradient_accumulation_steps 8
```

**Stage 2:**
```bash
# Use 4-bit quantization + smaller batch
python src/finetune_qwen3_8b_full.py \
  --load_in_4bit \
  --batch_size 4 \
  --gradient_accumulation_steps 32 \
  --gradient_checkpointing
```

### Slow Training

- Use `--dtype bfloat16` on supported hardware (A100, H100)
- Increase `--num_workers` for data loading
- Enable `--tf32` for NVIDIA Ampere GPUs
- Use multi-GPU with `torchrun` or `accelerate`

### W&B Login Issues

```bash
# Login to W&B
wandb login

# Or disable W&B
python src/finetune_qwen3_8b_vocab.py --disable_wandb
```

### Data Validation Errors

```bash
# Check data quality
python src/validate_training_data.py \
  --data-path data/Beauty_conversations_train_onerec.parquet

# Re-generate with smaller limits for testing
python src/generate_training_data_onerec.py \
  --limit-type-c 1000 \
  --limit-type-e 5000
```

---

## References

1. **semantic-ids-llm**: [github.com/marceloabk/semantic-ids-llm](https://github.com/marceloabk/semantic-ids-llm)
2. **OneRec-Think**: Original repository (this fork)
3. **Qwen3**: [Qwen/Qwen3-8B on HuggingFace](https://huggingface.co/Qwen/Qwen3-8B)
4. **Transformers**: [github.com/huggingface/transformers](https://github.com/huggingface/transformers)

---

## Citation

If you use this pipeline, please cite:

```bibtex
@article{onerec-think,
  title={OneRec-Think: Reasoning-Enhanced Recommender Systems},
  author={...},
  year={2024}
}

@article{semantic-ids-llm,
  title={Learning Semantic IDs with Large Language Models for Recommendation},
  author={...},
  year={2024}
}
```

---

## License

This pipeline inherits the licenses from OneRec-Think and semantic-ids-llm projects.
