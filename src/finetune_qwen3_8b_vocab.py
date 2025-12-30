#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for OneRec-Think semantic IDs.
Stage 1: Embedding initialization - trains only new token embeddings.

Adapted from semantic-ids-llm for OneRec-Think Beauty dataset.
"""

# Unsloth should be imported before trl, transformers, peft
from unsloth import FastLanguageModel, is_bfloat16_supported, add_new_tokens  # isort: skip

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from utils import DeviceManager, setup_logger, SYSTEM_PROMPT, REC_TEST_PROMPTS

logger = setup_logger("finetune-qwen3-vocab", log_to_file=True)


@dataclass
class FineTuneConfig:
    """Configuration for Stage 1: Embedding initialization."""

    # Model settings
    model_name: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # None for auto detection
    load_in_4bit: bool = False  # Must be False for embedding training
    random_state: int = 1368
    num_proc: int = 16
    enable_thinking: bool = False

    # Semantic ID vocabulary extension
    extend_vocabulary: bool = True
    codebook_levels: int = 4  # Number of hierarchical levels
    codebook_size: int = 256  # Number of codes per codebook
    num_semantic_tokens: int = 1024  # <|sid_0|> to <|sid_1023|>
    system_prompt: str = SYSTEM_PROMPT

    # Data settings
    category: str = "Beauty"
    data_dir: Path = Path("data")
    max_training_samples: int = 32000  # Sample size for embedding init

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 600
    num_train_epochs: int = 1
    warmup_steps: int = 60
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False

    # Optimizer settings
    optim: str = "adamw_8bit"

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_vocab_extended")
    steps_per_train_log: int = 60
    steps_per_val_log: int = 60
    save_steps: int = 5000
    deepspeed: Optional[str] = None

    # Computed paths
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.dtype is None:
            self.dtype = torch.float16 if not is_bfloat16_supported() else torch.bfloat16

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set data paths - OneRec format
        self.train_path = self.data_dir / f"{self.category}_conversations_train_onerec.parquet"
        self.val_path = self.data_dir / f"{self.category}_conversations_val_onerec.parquet"

        # Validate that training data exists
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please run src/prep_onerec_data.py and src/generate_training_data_onerec.py first."
            )

        # Validation data is optional
        if not self.val_path.exists():
            logger.warning(f"Validation data not found at {self.val_path}. Training without validation set.")

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=" * 80)
        logger.info("OneRec-Think Vocabulary Extension Configuration")
        logger.info("Stage 1: Embedding Initialization")
        logger.info("=" * 80)

        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  load_in_4bit: {self.load_in_4bit}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")

        logger.info("Vocabulary Extension:")
        logger.info(f"  extend_vocabulary: {self.extend_vocabulary}")
        logger.info(f"  codebook_levels: {self.codebook_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  num_semantic_tokens: {self.num_semantic_tokens}")
        logger.info(f"  Total new tokens: {self.num_semantic_tokens + 2}")

        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  max_training_samples: {self.max_training_samples}")

        logger.info("Training Parameters:")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  max_steps: {self.max_steps}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  optim: {self.optim}")
        logger.info(f"  deepspeed: {self.deepspeed}")
        logger.info("=" * 80)


def extend_tokenizer(model, tokenizer, config: FineTuneConfig):
    """Add semantic ID tokens to the tokenizer using Unsloth's add_new_tokens."""
    logger.info("=== Extending tokenizer with semantic ID tokens ===")

    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    original_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"Before - Vocab: {original_vocab_size:,}, Embedding: {original_embedding_size:,}, LM head: {original_lm_head_size:,}"
    )

    # Fix size mismatch if needed
    if original_embedding_size > original_vocab_size:
        logger.warning(f"⚠ Model has {original_embedding_size - original_vocab_size} more embeddings than tokenizer")
        model.resize_token_embeddings(original_vocab_size)
        original_embedding_size = model.get_input_embeddings().weight.shape[0]
        original_lm_head_size = model.get_output_embeddings().weight.shape[0]

    # Add special tokens
    new_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]

    # Add semantic ID tokens: <|sid_0|> through <|sid_1023|>
    for i in range(config.num_semantic_tokens):
        new_tokens.append(f"<|sid_{i}|>")

    logger.info(f"Adding {len(new_tokens)} new tokens")
    logger.info(f"  Special: <|rec|>, <|sid_start|>, <|sid_end|>")
    logger.info(f"  Semantic IDs: <|sid_0|> to <|sid_{config.num_semantic_tokens - 1}|>")

    # Add tokens using Unsloth
    add_new_tokens(model, tokenizer, new_tokens=new_tokens)

    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"After - Vocab: {new_vocab_size:,}, Embedding: {new_embedding_size:,}, LM head: {new_lm_head_size:,}"
    )

    # Verify consistency
    if new_vocab_size != new_embedding_size:
        logger.error(f"❌ CRITICAL: Tokenizer ({new_vocab_size}) != Embedding ({new_embedding_size})")
        model.resize_token_embeddings(new_vocab_size)
        new_embedding_size = model.get_input_embeddings().weight.shape[0]

    if new_vocab_size == new_embedding_size == new_lm_head_size:
        logger.info("✅ Model dimensions verified")
    else:
        logger.error("❌ Dimension mismatch persists!")

    num_added = new_vocab_size - original_vocab_size
    logger.info(f"✓ Successfully added {num_added} new tokens")
    logger.info("=" * 50)

    return num_added


def prepare_model(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """Prepare model for embedding-only training by freezing all except new embeddings."""
    logger.info("=== Preparing model for embedding-only training ===")

    current_vocab_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]

    logger.info(f"Vocab: {current_vocab_size:,}, New tokens: {num_new_tokens}")

    assert current_embedding_size == current_vocab_size, "Embedding size mismatch!"

    # Freeze all parameters
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze both input and output embeddings
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    embedding_layer.weight.requires_grad = True

    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True
        logger.info("✅ Unfroze input and output embedding layers")
    else:
        logger.error("❌ Could not access output embeddings!")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable: {trainable_params:,} / {total_params:,}")
    logger.info(f"Percentage: {100 * trainable_params / total_params:.4f}%")

    # Check new embedding initialization
    original_vocab_size = len(tokenizer) - num_new_tokens
    with torch.no_grad():
        new_embeddings = embedding_layer.weight[original_vocab_size:]
        logger.info("New embeddings statistics:")
        logger.info(f"  Shape: {new_embeddings.shape}")
        logger.info(f"  Mean: {new_embeddings.mean().item():.6f}")
        logger.info(f"  Std: {new_embeddings.std().item():.6f}")

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        model.gradient_checkpointing_disable()

    model.config.use_cache = not config.gradient_checkpointing

    logger.info("=== Model preparation complete ===")
    return model


def load_sid_dataset(config: FineTuneConfig, tokenizer, split="train"):
    """Load and prepare the conversation dataset with semantic IDs."""
    logger.info(f"Loading semantic ID conversation dataset ({split})")

    data_path = config.train_path if split == "train" else config.val_path
    logger.info(f"Loading from: {data_path}")

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} conversations")

    # Sample subset
    num_samples = min(len(dataset), 500 if split == "val" else config.max_training_samples)
    logger.info(f"Sampling {num_samples} examples for {split}")
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
    dataset = dataset.map(apply_chat_template, remove_columns=dataset.column_names, num_proc=config.num_proc)

    logger.info(f"Created dataset with {len(dataset)} examples")

    # Verify semantic IDs are present
    if split == "train" and len(dataset) > 0:
        sample_text = dataset[0]["text"]
        if "<|sid_start|>" in sample_text and "<|sid_end|>" in sample_text:
            logger.info("✓ Verified: Semantic ID tokens found in dataset")
            sid_count = sample_text.count("<|sid_start|>")
            logger.info(f"  Sample contains {sid_count} semantic ID(s)")
            logger.info("=" * 60)
            logger.info(f"Sample: {sample_text[:500]}...")
            logger.info("=" * 60)
        else:
            logger.warning("⚠ Warning: No semantic ID tokens found in sample")

    return dataset


class EmbeddingMonitorCallback(TrainerCallback):
    """Monitor embedding statistics during training."""

    def __init__(self, tokenizer, num_new_tokens, monitor_interval=100):
        self.tokenizer = tokenizer
        self.num_new_tokens = num_new_tokens
        self.monitor_interval = monitor_interval
        self.original_vocab_size = len(tokenizer) - num_new_tokens
        self.initial_embeddings = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        embeddings = model.get_input_embeddings().weight
        self.initial_embeddings = embeddings[self.original_vocab_size:].clone().detach()

        mean = self.initial_embeddings.mean().item()
        std = self.initial_embeddings.std().item()

        wandb.log({
            "embeddings/initial_mean": mean,
            "embeddings/initial_std": std,
        })

        logger.info(f"Initial embeddings - Mean: {mean:.4f}, Std: {std:.4f}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.monitor_interval == 0 and state.global_step > 0:
            embeddings = model.get_input_embeddings().weight
            new_embeddings = embeddings[self.original_vocab_size:]

            change_from_init = (new_embeddings - self.initial_embeddings).abs().mean().item()
            mean = new_embeddings.mean().item()
            std = new_embeddings.std().item()

            wandb.log({
                "embeddings/change_from_init": change_from_init,
                "embeddings/mean": mean,
                "embeddings/std": std,
                "step": state.global_step,
            })

            logger.info(
                f"Step {state.global_step} - Change: {change_from_init:.4f}, "
                f"Mean: {mean:.4f}, Std: {std:.4f}"
            )


class SemanticIDGenerationCallback(TrainerCallback):
    """Test semantic ID generation periodically."""

    def __init__(self, tokenizer, test_interval=200):
        self.tokenizer = tokenizer
        self.test_interval = test_interval
        self.test_messages = REC_TEST_PROMPTS

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
        results = []

        for i, messages in enumerate(self.test_messages[:3], 1):  # Test first 3 only
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.8)

                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):]

                has_sids = "<|sid_start|>" in generated or "<|sid_end|>" in generated
                if has_sids:
                    successful += 1

                user_msg = messages[-1]["content"]
                results.append([step, user_msg[:50], generated[:100], has_sids])

                logger.info(f"\nTest {i}: {user_msg[:50]}...")
                logger.info(f"  Generated: {generated[:100]}...")
                logger.info(f"  Has SIDs: {has_sids}")

            except Exception as e:
                logger.warning(f"Generation failed: {e}")

        success_rate = successful / min(3, len(self.test_messages))

        wandb.log({
            "generation/success_rate": success_rate,
            "generation/successful_count": successful,
            "step": step,
        })

        logger.info(f"\nSuccess rate: {success_rate:.0%}")
        model.train(training_mode)
        logger.info("=" * 60)


def train_embeddings(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """Train only the new token embeddings using SFTTrainer."""
    logger.info("Starting embedding initialization training")

    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val") if config.val_path.exists() else None

    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(val_dataset) if val_dataset else 0,
        "dataset/vocabulary_size": len(tokenizer),
        "dataset/new_tokens": num_new_tokens,
    })

    # SFT configuration
    sft_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=config.num_proc,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.steps_per_train_log,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.random_state,
        output_dir=str(config.output_dir),
        save_steps=config.save_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",
        save_strategy="steps",
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.steps_per_val_log if val_dataset else None,
        per_device_eval_batch_size=config.batch_size,
        save_total_limit=2,
        deepspeed=config.deepspeed,
    )

    callbacks = [
        EmbeddingMonitorCallback(tokenizer, num_new_tokens, monitor_interval=config.steps_per_val_log),
        SemanticIDGenerationCallback(tokenizer, test_interval=config.steps_per_val_log),
    ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    # Log GPU stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
        logger.info(f"Reserved: {start_memory} GB")

    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(f"Training time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")
        logger.info(f"Peak reserved memory: {used_memory} GB")

    wandb.summary["total_steps"] = trainer_stats.global_step if hasattr(trainer_stats, "global_step") else config.max_steps

    logger.info("Embedding initialization completed!")
    return trainer_stats


def save_model_and_tokenizer(model, tokenizer, config: FineTuneConfig):
    """Save the model with initialized embeddings and extended tokenizer."""
    logger.info("Saving model and tokenizer")

    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    logger.info(f"Input: {input_size}, Output: {output_size}, Tokenizer: {vocab_size}")

    if input_size != vocab_size or output_size != vocab_size:
        logger.error("❌ Size mismatch before save!")
        model.resize_token_embeddings(vocab_size)

    assert input_size == vocab_size and output_size == vocab_size, "Dimension mismatch!"
    logger.info("✅ Dimensions verified")

    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))

    logger.info(f"Saved to: {final_save_path}")

    config_dict = {
        "stage": "vocab_extension",
        "model_name": config.model_name,
        "num_semantic_tokens": config.num_semantic_tokens,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
    }

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    parser = HfArgumentParser(FineTuneConfig)
    (config,) = parser.parse_args_into_dataclasses(return_remaining_strings=True)[:1]
    device_manager = DeviceManager(logger)

    run_name = f"qwen3-vocab-{config.category}-lr{config.learning_rate}"
    wandb.init(project="onerec-semantic-id-vocab", name=run_name, config=config.__dict__)
    config.log_config()

    logger.info("Loading base model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    num_new_tokens = 0
    if config.extend_vocabulary:
        num_new_tokens = extend_tokenizer(model, tokenizer, config)
        model = prepare_model(model, tokenizer, config, num_new_tokens)

    train_stats = train_embeddings(model, tokenizer, config, num_new_tokens)

    logger.info("Saving embeddings as W&B artifact")
    embeddings = model.get_input_embeddings().weight
    new_embeddings = embeddings[len(tokenizer) - num_new_tokens:].detach().cpu()

    artifact = wandb.Artifact(
        f"semantic_embeddings_{config.category}",
        type="embeddings",
        description=f"Trained semantic ID embeddings for {config.category}",
    )

    embeddings_path = config.output_dir / "semantic_embeddings.npy"
    np.save(embeddings_path, new_embeddings.float().numpy())
    artifact.add_file(str(embeddings_path))
    wandb.log_artifact(artifact)

    save_model_and_tokenizer(model, tokenizer, config)

    wandb.finish()

    logger.info("=" * 80)
    logger.info("Stage 1: Embedding initialization complete!")
    logger.info(f"Initialized {num_new_tokens} new semantic ID tokens")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 80)
