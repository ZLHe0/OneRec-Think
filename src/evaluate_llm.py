#!/usr/bin/env python3
"""
Evaluate trained OneRec-Think LLM for next-item prediction
Calculates Hit@K and NDCG@K metrics using beam search generation

Adapted from OneRec-Think/test/test_model_hitrate.py for our training format
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

from utils import setup_logger, SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OneRec-Think LLM")

    # Model settings
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (Stage 1 or Stage 2)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")

    # Data settings
    parser.add_argument("--data_path", type=str,
                        default="data/Beauty_sequences_val.parquet",
                        help="Path to val/test sequences parquet file (default: val split)")
    parser.add_argument("--split_type", type=str, choices=["val", "test"], default="val",
                        help="Split type: 'val' for validation (has targets), 'test' for prediction-only")
    parser.add_argument("--min_history", type=int, default=1,
                        help="Minimum history length for evaluation (default: 1)")

    # Generation settings
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_beams", type=int, default=10,
                        help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")

    # Evaluation settings
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="Metrics to calculate (comma-separated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output settings
    parser.add_argument("--log_file", type=str, default="logs/evaluation.log",
                        help="Log file path")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Save detailed results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print generation examples")

    # W&B settings
    parser.add_argument("--wandb_project", type=str, default="onerec-llm-evaluation",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="W&B tags (comma-separated)")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable W&B logging")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceEvalDataset(Dataset):
    """
    Dataset for sequence evaluation using train/val/test splits

    Expects DataFrame with columns:
    - semantic_id_sequence: List of semantic IDs (history)
    - target_sid: Target semantic ID (None for test split)
    - user_id: User ID
    - sequence_length: Length of history
    """

    def __init__(self, sequences_df: pd.DataFrame, min_history: int = 1, split_type: str = "val", logger=None):
        self.logger = logger
        self.split_type = split_type
        self.data = []

        # Process sequences
        for _, row in sequences_df.iterrows():
            history = row['semantic_id_sequence']  # Already pre-split in preprocessing
            target = row.get('target_sid', None)
            seq_len = row['sequence_length']
            user_id = row.get('user_id', 'unknown')

            # Skip sequences with insufficient history
            if seq_len < min_history:
                continue

            # For test split, target may be None (real-world prediction)
            if split_type == "val" and target is None:
                if logger:
                    logger.warning(f"Validation sample with no target - skipping")
                continue

            self.data.append({
                'user_id': user_id,
                'history': history,
                'target': target,
                'history_length': len(history)
            })

        if logger:
            logger.info(f"Created {split_type} dataset with {len(self.data)} evaluation samples")
            hist_lengths = [d['history_length'] for d in self.data]
            if hist_lengths:
                logger.info(f"History length stats: min={min(hist_lengths)}, max={max(hist_lengths)}, avg={np.mean(hist_lengths):.1f}")
            if split_type == "test":
                num_with_targets = sum(1 for d in self.data if d['target'] is not None)
                logger.info(f"Samples with targets: {num_with_targets}/{len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_evaluation_prompt(history: List[str], system_prompt: str = SYSTEM_PROMPT) -> List[Dict]:
    """
    Create evaluation prompt in the same format as training Type C data

    Args:
        history: List of semantic IDs representing purchase history
        system_prompt: System prompt to use

    Returns:
        List of message dicts for chat template
    """
    # Use last 3 items (or all if less than 3)
    last_n = history[-3:] if len(history) >= 3 else history
    history_text = ' '.join(last_n)

    # Use same format as training Type C data
    user_message = f"Based on this purchase history: {history_text}, predict the next item."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def extract_sid_from_response(text: str) -> str:
    """
    Extract semantic ID from model response

    Looks for pattern: <|sid_start|><|sid_X|><|sid_Y|><|sid_Z|><|sid_W|><|sid_end|>
    """
    # Pattern for our SID format
    sid_pattern = r'<\|sid_start\|>(<\|sid_\d+\|>){4}<\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()


def calculate_hit_k(topk_results: List[List[int]], k: int) -> float:
    """Calculate hit@k metric"""
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results) if topk_results else 0.0


def calculate_ndcg_k(topk_results: List[List[int]], k: int) -> float:
    """Calculate NDCG@k metric"""
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Best case: hit at position 1
        ndcg += dcg / idcg
    return ndcg / len(topk_results) if topk_results else 0.0


def evaluate_batch(
    model,
    tokenizer,
    batch: Dict,
    num_beams: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logger=None,
    verbose: bool = False
) -> Tuple[List[List[int]], List[str], List[float]]:
    """
    Evaluate a batch of sequences

    Returns:
        topk_results: List of binary lists indicating hits for each beam
        predictions: List of generated predictions
        scores: List of generation scores
    """
    # Create prompts
    prompts = []
    targets = batch['target']

    for i in range(len(targets)):
        messages = create_evaluation_prompt(batch['history'][i])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate with beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            temperature=temperature,
            top_p=top_p,
        )

    # Decode outputs
    output_ids = outputs['sequences']
    scores = outputs.get('sequences_scores', None)

    generated_texts = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    # Process scores
    if scores is not None:
        scores_list = [float(s) for s in scores.detach().cpu().tolist()]
    else:
        scores_list = [0.0] * len(generated_texts)

    # Extract predictions and calculate topk results
    topk_results = []
    batch_predictions = []

    batch_size = len(targets)
    for b in range(batch_size):
        # Get beams for this sample
        start_idx = b * num_beams
        end_idx = start_idx + num_beams

        beams = generated_texts[start_idx:end_idx]
        beam_scores = scores_list[start_idx:end_idx]

        # Extract predictions from each beam
        predictions = []
        for beam_text in beams:
            # Extract text after the prompt
            if len(prompts[b]) < len(beam_text):
                response = beam_text[len(prompts[b]):]
                pred_sid = extract_sid_from_response(response)
            else:
                pred_sid = extract_sid_from_response(beam_text)
            predictions.append(pred_sid)

        batch_predictions.append(predictions)

        # Calculate hits
        target_sid = targets[b]
        hits = [1 if pred == target_sid else 0 for pred in predictions]
        topk_results.append(hits)

        # Verbose logging
        if verbose and logger and b == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Sample {b}:")
            logger.info(f"History: {' '.join(batch['history'][b][-3:])}")
            logger.info(f"Target: {target_sid}")
            logger.info(f"Predictions:")
            for i, (pred, score, hit) in enumerate(zip(predictions, beam_scores, hits)):
                logger.info(f"  Rank {i+1} [score={score:.4f}]: {pred} {'✓' if hit else '✗'}")
            logger.info('='*60)

    return topk_results, batch_predictions, scores_list


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup logging
    logger = setup_logger("evaluate-llm", log_to_file=True, log_dir=Path(args.log_file).parent)

    logger.info("="*80)
    logger.info("OneRec-Think LLM Evaluation")
    logger.info("="*80)
    logger.info(f"\nNote: Using leave-last-out protocol aligned with OneRec-Think")
    logger.info(f"  - Val split: items[:-1] → item[-1]")
    logger.info(f"  - Test split: items[:] → next item (no ground truth)")
    logger.info(f"\nEvaluation split: {args.split_type}")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize W&B
    if not args.disable_wandb:
        # Generate run name if not provided
        if args.wandb_run_name is None:
            model_name = Path(args.model_path).name
            run_name = f"eval-{model_name}-beams{args.num_beams}"
        else:
            run_name = args.wandb_run_name

        # Parse tags
        tags = args.wandb_tags.split(",") if args.wandb_tags else ["evaluation"]

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=tags
        )
        logger.info(f"W&B initialized: project={args.wandb_project}, run={run_name}")
    else:
        logger.info("W&B logging disabled")

    # Load model and tokenizer
    logger.info(f"\nLoading model from: {args.model_path}")
    tokenizer_path = args.tokenizer_path or args.model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")
    logger.info(f"Vocabulary size: {len(tokenizer)}")

    # Log model info to W&B
    if not args.disable_wandb:
        wandb.config.update({
            "model_device": str(next(model.parameters()).device),
            "vocabulary_size": len(tokenizer),
            "model_parameters": sum(p.numel() for p in model.parameters())
        })

    # Load data
    logger.info(f"\nLoading {args.split_type} sequences from: {args.data_path}")
    eval_df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(eval_df)} sequences from {args.split_type} split")

    # Create dataset
    eval_dataset = SequenceEvalDataset(
        eval_df,
        min_history=args.min_history,
        split_type=args.split_type,
        logger=logger
    )

    # Log dataset info to W&B
    if not args.disable_wandb:
        wandb.config.update({
            "num_eval_sequences": len(eval_dataset),
            "total_sequences_in_file": len(eval_df),
            "split_type": args.split_type,
            "min_history": args.min_history
        })

    # Create dataloader
    def collate_fn(batch):
        return {
            'user_id': [b['user_id'] for b in batch],
            'history': [b['history'] for b in batch],
            'target': [b['target'] for b in batch],
            'history_length': [b['history_length'] for b in batch]
        }

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluate
    logger.info(f"\nStarting evaluation...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num beams: {args.num_beams}")

    all_topk_results = []
    all_predictions = []

    # Track progress
    total_batches = len(eval_loader)
    log_interval = max(1, total_batches // 20)  # Log 20 times during evaluation

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating {args.split_type} split")):
        topk_results, predictions, scores = evaluate_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            logger=logger,
            verbose=args.verbose and batch_idx == 0
        )

        all_topk_results.extend(topk_results)
        all_predictions.extend(predictions)

        # Log intermediate metrics to W&B
        if not args.disable_wandb and (batch_idx + 1) % log_interval == 0:
            samples_processed = len(all_topk_results)
            progress = (batch_idx + 1) / total_batches

            # Calculate intermediate metrics
            interim_metrics = {
                "progress": progress,
                "samples_processed": samples_processed,
                "batches_processed": batch_idx + 1
            }

            # Calculate interim Hit@1 and NDCG@10 for tracking
            if all_topk_results:
                interim_metrics["interim/hit@1"] = calculate_hit_k(all_topk_results, 1)
                interim_metrics["interim/hit@10"] = calculate_hit_k(all_topk_results, 10)
                interim_metrics["interim/ndcg@10"] = calculate_ndcg_k(all_topk_results, 10)

            wandb.log(interim_metrics, step=batch_idx)

            logger.info(f"Progress: {progress*100:.1f}% | Samples: {samples_processed} | "
                       f"Hit@1: {interim_metrics.get('interim/hit@1', 0):.4f} | "
                       f"NDCG@10: {interim_metrics.get('interim/ndcg@10', 0):.4f}")

    # Calculate metrics
    logger.info("\n" + "="*80)
    logger.info("Evaluation Results")
    logger.info("="*80)

    # Check if we can calculate metrics (need targets)
    num_samples_with_targets = sum(1 for results_list in all_topk_results if any(r is not None for r in results_list))

    if args.split_type == "test" and num_samples_with_targets == 0:
        logger.warning("Test split has no ground truth targets - cannot calculate metrics")
        logger.info(f"Generated predictions for {len(all_predictions)} samples")
        results = {}
    else:
        metrics_list = args.metrics.split(",")
        results = {}

        for metric in metrics_list:
            metric = metric.strip().lower()
            if metric.startswith("hit@"):
                k = int(metric.split("@")[1])
                value = calculate_hit_k(all_topk_results, k)
                results[f"hit@{k}"] = value
                logger.info(f"Hit@{k:2d}:  {value:.4f} ({value*100:.2f}%)")
            elif metric.startswith("ndcg@"):
                k = int(metric.split("@")[1])
                value = calculate_ndcg_k(all_topk_results, k)
                results[f"ndcg@{k}"] = value
                logger.info(f"NDCG@{k:2d}: {value:.4f}")

    logger.info("="*80)

    # Log final metrics to W&B
    if not args.disable_wandb:
        if results:
            # Create final metrics dict for W&B
            final_metrics = {f"final/{k}": v for k, v in results.items()}
            final_metrics["final/num_samples"] = len(all_topk_results)

            wandb.log(final_metrics)

            # Log summary metrics
            wandb.summary.update(results)
            wandb.summary["num_samples"] = len(all_topk_results)

            # Create a comparison table
            metrics_table = wandb.Table(
                columns=["Metric", "Value", "Percentage"],
                data=[
                    [k, v, f"{v*100:.2f}%" if k.startswith("hit@") else "N/A"]
                    for k, v in results.items()
                ]
            )
            wandb.log({"metrics_table": metrics_table})

            logger.info("Final metrics logged to W&B")
        else:
            # No metrics for test split without targets
            wandb.summary["num_predictions"] = len(all_predictions)
            wandb.summary["split_type"] = args.split_type
            logger.info("Prediction counts logged to W&B (no metrics available)")

    # Save results
    if args.save_results:
        results_data = {
            "args": vars(args),
            "metrics": results,
            "num_samples": len(all_topk_results)
        }

        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"\nResults saved to: {save_path}")

    logger.info("\n✅ Evaluation completed successfully!")

    # Finish W&B run
    if not args.disable_wandb:
        wandb.finish()
        logger.info("W&B run finished")

    return results


if __name__ == "__main__":
    main()
