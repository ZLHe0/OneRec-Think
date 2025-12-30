#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for OneRec-Think semantic IDs.
Stage 2: Full fine-tuning - trains all model parameters.

Adapted from semantic-ids-llm for OneRec-Think Beauty dataset.
"""

# Unsloth should be imported before trl, transformers, peft
from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip

import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import HfArgumentParser, TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from utils import DeviceManager, setup_logger, SYSTEM_PROMPT, REC_TEST_PROMPTS

logger = setup_logger("finetune-qwen3-full", log_to_file=True)


@dataclass
class FullFineTuneConfig:
    """Configuration for Stage 2: Full fine-tuning."""

    # Model settings - Load from vocab extension checkpoint
    model_name: str = "models/qwen3_8b_vocab_extended/final"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False
    random_state: int = 1368
    num_proc: int = 16
    enable_thinking: bool = False

    # System prompt
    system_prompt: str = SYSTEM_PROMPT

    # Data settings
    category: str = "Beauty"
    data_dir: Path = Path("data")
    use_full_dataset: bool = True
    max_training_samples: Optional[int] = None  # None = use all data

    # Full finetuning parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2 # (8 GPU 8 -> 2)
    gradient_clip_norm: float = 1.0
    max_steps: int = -1  # -1 = use num_train_epochs
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False
    optim: str = "adamw_8bit"

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_full_finetuned")
    logging_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_samples: int = 10000
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    deepspeed: Optional[str] = None

    # Checkpoint resume
    resume_from_checkpoint: bool = False

    # Computed paths
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.dtype is None:
            self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set data paths - OneRec format
        self.train_path = self.data_dir / f"{self.category}_conversations_train_onerec.parquet"
        self.val_path = self.data_dir / f"{self.category}_conversations_val_onerec.parquet"

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please run data generation scripts first."
            )

        if not self.val_path.exists():
            logger.warning(f"Validation data not found at {self.val_path}")
            self.eval_strategy = "no"
            self.load_best_model_at_end = False

        # Verify Stage 1 checkpoint exists
        stage1_checkpoint = Path(self.model_name)
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {stage1_checkpoint}. "
                f"Please run finetune_qwen3_8b_vocab.py first."
            )

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=" * 80)
        logger.info("OneRec-Think Full Fine-tuning Configuration")
        logger.info("Stage 2: Full Model Training")
        logger.info("=" * 80)

        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")

        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  use_full_dataset: {self.use_full_dataset}")

        logger.info("Training Parameters:")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  warmup_ratio: {self.warmup_ratio}")
        logger.info(f"  optim: {self.optim}")
        logger.info(f"  deepspeed: {self.deepspeed}")

        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  eval_strategy: {self.eval_strategy}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info("=" * 80)


def get_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in the output directory."""
    checkpoint_pattern = str(output_dir / "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    if not checkpoint_dirs:
        logger.info("No existing checkpoints found")
        return None

    try:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split("-")[-1]))
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing checkpoint directories: {e}")
        return None


def load_vocab_extended_model(config: FullFineTuneConfig):
    """Load the model from Stage 1 checkpoint with extended vocabulary."""
    logger.info(f"Loading Stage 1 checkpoint from: {config.model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    vocab_size = len(tokenizer)
    logger.info(f"Loaded model with vocabulary size: {vocab_size}")

    # Verify semantic ID tokens
    test_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>", "<|sid_0|>", "<|sid_1023|>"]
    for token in test_tokens:
        if token in tokenizer.get_vocab():
            logger.info(f"✓ Found token: {token}")
        else:
            logger.warning(f"⚠ Missing token: {token}")

    # Test tokenization
    logger.info("Testing semantic ID tokenization")
    test_string = "<|rec|><|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|>"
    token_ids = tokenizer.encode(test_string, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    logger.info(f"Test string: {test_string}")
    logger.info(f"Tokens: {tokens}")

    # Verify round-trip
    decoded_string = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert decoded_string == test_string, f"Round-trip mismatch!"
    logger.info("✓ Round-trip encoding verified")

    # Configure gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        model.gradient_checkpointing_disable()

    # Ensure all parameters are trainable
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
            if not param.requires_grad:
                param.requires_grad = True
            trainable_params += param.numel()

    logger.info(f"Trainable: {trainable_params:,} / {total_params:,}")
    logger.info(f"Percentage: {100 * trainable_params / total_params:.2f}%")

    # Verify model dimensions
    logger.info("=== Model Verification ===")
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]

    logger.info(f"Input embedding size: {input_size}")
    logger.info(f"Output embedding size: {output_size}")
    logger.info(f"Tokenizer vocab size: {vocab_size}")

    if output_size != vocab_size:
        logger.error(f"❌ CRITICAL: Output ({output_size}) != Vocab ({vocab_size})")
        logger.info("Attempting to resize model")
        model.resize_token_embeddings(vocab_size)
        new_output_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After resize - Output: {new_output_size}")

        if new_output_size == vocab_size:
            logger.info("✅ Model successfully resized")
        else:
            logger.error("❌ Resize failed!")
    else:
        logger.info("✅ Model dimensions verified")

    logger.info("=" * 50)

    return model, tokenizer


def load_sid_dataset(config: FullFineTuneConfig, tokenizer, split="train"):
    """Load the full conversation dataset with semantic IDs."""
    logger.info(f"Loading semantic ID dataset ({split})")

    data_path = config.train_path if split == "train" else config.val_path

    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}")
        return None

    logger.info(f"Loading from: {data_path}")

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset):,} conversations")

    # Apply sampling
    if not config.use_full_dataset and config.max_training_samples and split == "train":
        num_samples = min(len(dataset), config.max_training_samples)
        logger.info(f"Sampling {num_samples:,} examples")
        dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))
    elif split == "val":
        num_samples = min(len(dataset), config.eval_samples)
        logger.info(f"Using {num_samples:,} validation examples")
        dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    # Apply chat template
    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    logger.info("Applying chat template")
    dataset = dataset.map(
        apply_chat_template,
        remove_columns=dataset.column_names,
        num_proc=config.num_proc,
        batch_size=1000,
    )

    logger.info(f"Prepared dataset with {len(dataset):,} examples")

    # Verify semantic IDs
    if len(dataset) > 0:
        sample_text = dataset[0]["text"]
        sid_count = sample_text.count("<|sid_start|>")
        if sid_count > 0:
            logger.info(f"✓ Verified: Found {sid_count} semantic IDs in sample")
        else:
            logger.warning("⚠ No semantic IDs found in sample")

        logger.info("=" * 60)
        logger.info(f"Sample ({split}): {sample_text[:500]}...")
        logger.info("=" * 60)

    return dataset


class ModelMonitorCallback(TrainerCallback):
    """Monitor training progress and log to W&B."""

    def __init__(self, config, monitor_interval=100):
        self.config = config
        self.monitor_interval = monitor_interval
        self.initial_loss = None
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics."""
        if logs and "loss" in logs:
            if self.initial_loss is None:
                self.initial_loss = logs["loss"]

            current_loss = logs["loss"]
            improvement = (self.initial_loss - current_loss) / self.initial_loss * 100 if self.initial_loss else 0

            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)
            epoch = logs.get("epoch", 0)

            current_time = time.time()
            time_elapsed = current_time - self.last_log_time
            steps_done = state.global_step - self.last_log_step

            effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            samples_processed = steps_done * effective_batch_size
            samples_per_second = samples_processed / time_elapsed if time_elapsed > 0 else 0

            self.last_log_time = current_time
            self.last_log_step = state.global_step

            log_str = (
                f"Step {state.global_step:05d} | Epoch {epoch:.2f} | "
                f"lr: {lr:.2e} | loss: {current_loss:.4f} | "
                f"improvement: {improvement:+.1f}% | samples/s: {samples_per_second:.0f}"
            )

            logger.info(log_str)

            wandb.log({
                "loss/train": current_loss,
                "metrics/learning_rate": lr,
                "metrics/gradient_norm": grad_norm,
                "metrics/improvement_pct": improvement,
                "metrics/samples_per_second": samples_per_second,
            }, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)
            logger.info(f"EVAL | Step {state.global_step:05d} | loss: {eval_loss:.4f}")

            wandb.log({
                "loss/eval": eval_loss,
            }, step=state.global_step)


class GenerationTestCallback(TrainerCallback):
    """Test generation periodically during training."""

    def __init__(self, tokenizer, test_interval=1000):
        self.tokenizer = tokenizer
        self.test_interval = test_interval
        self.test_messages = REC_TEST_PROMPTS[:3]  # Use first 3 test prompts

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self.test_generation(model, state.global_step)

    def test_generation(self, model, step):
        logger.info("=" * 60)
        logger.info(f"Testing generation at step {step}")
        logger.info("=" * 60)

        training_mode = model.training
        model.eval()

        successful = 0

        for i, messages in enumerate(self.test_messages, 1):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.8)

                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):]

                has_sids = "<|sid_start|>" in generated
                if has_sids:
                    successful += 1

                user_msg = messages[-1]["content"]
                logger.info(f"\nTest {i}: {user_msg[:50]}...")
                logger.info(f"  Generated: {generated[:100]}...")
                logger.info(f"  Has SIDs: {has_sids}")

            except Exception as e:
                logger.warning(f"Generation failed: {e}")

        success_rate = successful / len(self.test_messages)
        logger.info(f"\nSuccess rate: {success_rate:.0%}")

        wandb.log({
            "generation/success_rate": success_rate,
            "step": step,
        })

        model.train(training_mode)
        logger.info("=" * 60)


def train_full_model(model, tokenizer, config: FullFineTuneConfig):
    """Train the full model using SFTTrainer."""
    logger.info("Starting full model fine-tuning")

    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val") if config.val_path.exists() else None

    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(val_dataset) if val_dataset else 0,
        "dataset/vocabulary_size": len(tokenizer),
    })

    # SFT configuration
    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        num_train_epochs=config.num_train_epochs if config.max_steps <= 0 else 1,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.gradient_clip_norm,
        seed=config.random_state,
        output_dir=str(config.output_dir),
        save_steps=config.save_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",
        save_strategy=config.save_strategy,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy=config.eval_strategy if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        per_device_eval_batch_size=config.batch_size,
        metric_for_best_model=config.metric_for_best_model if val_dataset else None,
        greater_is_better=config.greater_is_better,
        load_best_model_at_end=config.load_best_model_at_end if val_dataset else False,
        save_total_limit=config.save_total_limit,
        deepspeed=config.deepspeed,
    )

    callbacks = [
        ModelMonitorCallback(config, monitor_interval=config.logging_steps),
        GenerationTestCallback(tokenizer, test_interval=config.eval_steps),
    ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # GPU stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
        logger.info(f"Reserved: {start_memory} GB")

    # Check for checkpoint resume
    resume_from = None
    if config.resume_from_checkpoint:
        resume_from = get_latest_checkpoint(config.output_dir)

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(f"Training time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")
        logger.info(f"Peak memory: {used_memory} GB")

    logger.info("Full fine-tuning completed!")
    return trainer_stats


def save_final_model(model, tokenizer, config: FullFineTuneConfig):
    """Save the final trained model."""
    logger.info("Saving final model")

    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))

    logger.info(f"Saved to: {final_save_path}")

    config_dict = {
        "stage": "full_finetuning",
        "model_name": config.model_name,
        "category": config.category,
        "num_epochs": config.num_train_epochs,
        "vocabulary_size": len(tokenizer),
    }

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training configuration saved")


if __name__ == "__main__":
    parser = HfArgumentParser(FullFineTuneConfig)
    (config,) = parser.parse_args_into_dataclasses(return_remaining_strings=True)[:1]
    device_manager = DeviceManager(logger)

    run_name = f"qwen3-full-{config.category}-lr{config.learning_rate}"
    wandb.init(project="onerec-semantic-id-full", name=run_name, config=config.__dict__)
    config.log_config()

    logger.info("Loading Stage 1 model")
    model, tokenizer = load_vocab_extended_model(config)

    train_stats = train_full_model(model, tokenizer, config)

    save_final_model(model, tokenizer, config)

    wandb.finish()

    logger.info("=" * 80)
    logger.info("Stage 2: Full fine-tuning complete!")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 80)
