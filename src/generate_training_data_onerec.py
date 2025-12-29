#!/usr/bin/env python3
"""
OneRec-Think LLM Training Data Generation

Generates ~1.1M ChatML-formatted training conversations from OneRec-Think Beauty dataset.
Based on semantic-ids-llm methodology with 5 types of training tasks:
- Type A: Semantic ID → Text mappings (4 variations)
- Type B: Text → Semantic ID mappings (6 variations)
- Type C: Sequential prediction (next-item from history)
- Type D: Semantic understanding (hierarchical structure)
- Type E: Multi-hop reasoning (co-purchase patterns)
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import json


SYSTEM_PROMPT = """You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_X|><|sid_Y|><|sid_Z|><|sid_W|><|sid_end|> that encode product relationships and categories."""


def create_conversation(input_text: str, output_text: str, sample_type: str) -> Dict:
    """Create a ChatML-formatted conversation."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ],
        "type": sample_type
    }


def generate_type_a_data(items_df: pd.DataFrame) -> List[Dict]:
    """
    Type A: Semantic ID → Text mappings
    4 variations: sid_to_title, sid_to_description, sid_to_category, sid_to_context
    Expected: ~48,400 samples (12,101 × 4)
    """
    conversations = []

    for _, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Type A"):
        sid = row['semantic_id']
        title = row['title']
        description = row['description_text']
        categories = row['categories_text']
        context = row['item_context']

        # A1: sid_to_title
        conversations.append(create_conversation(
            f"What is the title of the product with semantic ID {sid}?",
            title,
            "type_a_sid_to_title"
        ))

        # A2: sid_to_description
        conversations.append(create_conversation(
            f"What is the description of the product {sid}?",
            description,
            "type_a_sid_to_description"
        ))

        # A3: sid_to_category
        conversations.append(create_conversation(
            f"What category does the product {sid} belong to?",
            categories,
            "type_a_sid_to_category"
        ))

        # A4: sid_to_context (full metadata)
        conversations.append(create_conversation(
            f"Tell me about the product {sid}.",
            context,
            "type_a_sid_to_context"
        ))

    return conversations


def generate_type_b_data(items_df: pd.DataFrame) -> List[Dict]:
    """
    Type B: Text → Semantic ID mappings
    6 variations: title/description/category queries with different matching strategies
    Expected: ~72,600 samples (12,101 × 6)
    """
    conversations = []

    for _, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Type B"):
        sid = row['semantic_id']
        title = row['title']
        description = row['description_text']
        categories = row['categories_text']

        # B1: title_to_sid (exact match)
        conversations.append(create_conversation(
            f"What is the semantic ID for the product titled '{title}'?",
            sid,
            "type_b_title_to_sid"
        ))

        # B2: title_prefix_to_sid
        title_words = title.split()
        if len(title_words) >= 3:
            prefix = ' '.join(title_words[:3])
            conversations.append(create_conversation(
                f"What is the semantic ID for the product starting with '{prefix}'?",
                sid,
                "type_b_title_prefix_to_sid"
            ))

        # B3: title_contains_to_sid (use middle portion)
        if len(title_words) >= 5:
            middle = ' '.join(title_words[1:4])
            conversations.append(create_conversation(
                f"Find the semantic ID for a product whose title contains '{middle}'.",
                sid,
                "type_b_title_contains_to_sid"
            ))

        # B4: description_to_sid (first 100 chars)
        desc_snippet = description[:100] if len(description) > 100 else description
        conversations.append(create_conversation(
            f"What product has this description: '{desc_snippet}'?",
            sid,
            "type_b_description_to_sid"
        ))

        # B5: description_contains_to_sid (random snippet)
        if len(description) > 50:
            start_idx = random.randint(0, max(0, len(description) - 50))
            snippet = description[start_idx:start_idx + 50]
            conversations.append(create_conversation(
                f"Find the product with description containing '{snippet}'.",
                sid,
                "type_b_description_contains_to_sid"
            ))

        # B6: categories_to_sid
        conversations.append(create_conversation(
            f"What is the semantic ID for a product in category '{categories}'?",
            sid,
            "type_b_categories_to_sid"
        ))

    return conversations


def generate_type_c_data(sequences_df: pd.DataFrame) -> List[Dict]:
    """
    Type C: Sequential prediction (next-item from user history)
    Uses training split with pre-determined targets (items[:-2] → item[-2])

    Note: sequences_df should be the training split with target_sid column
    Expected: ~223,600 samples (~10 per sequence)
    """
    conversations = []

    for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc="Type C"):
        history = row['semantic_id_sequence']  # Already items[:-2] from training split
        target = row['target_sid']  # item[-2]
        seq_len = row['sequence_length']

        # C1: Predict from last 2 items (if available)
        if seq_len >= 2:
            last_2 = ' '.join(history[-2:])
            conversations.append(create_conversation(
                f"A user purchased these items in sequence: {last_2}. What might they buy next?",
                target,
                "type_c_seq_last_2"
            ))

        # C2: Predict from last 3 items (if available)
        if seq_len >= 3:
            last_3 = ' '.join(history[-3:])
            conversations.append(create_conversation(
                f"Based on this purchase history: {last_3}, predict the next item.",
                target,
                "type_c_seq_last_3"
            ))

        # C3: Predict from last 5 items (if available)
        if seq_len >= 5:
            last_5 = ' '.join(history[-5:])
            conversations.append(create_conversation(
                f"Given the purchase sequence {last_5}, what is the likely next purchase?",
                target,
                "type_c_seq_last_5"
            ))

        # C4: Predict from full history
        full_history = ' '.join(history)
        conversations.append(create_conversation(
            f"A user has the following purchase history: {full_history}. What should we recommend next?",
            target,
            "type_c_seq_full"
        ))

    return conversations


def generate_type_d_data(items_df: pd.DataFrame) -> List[Dict]:
    """
    Type D: Semantic understanding (hierarchical structure)
    Queries about semantic ID prefixes, hierarchies, and similarity
    Expected: ~11,000 samples
    """
    conversations = []

    # Group by semantic ID prefixes
    items_df['sid_level_0'] = items_df['semantic_id_0']
    items_df['sid_level_1'] = items_df['semantic_id_1']
    items_df['sid_prefix_2'] = items_df['semantic_id_0'].astype(str) + '_' + items_df['semantic_id_1'].astype(str)

    # D1: Prefix to category
    for (sid_0, sid_1), group in items_df.groupby(['sid_level_0', 'sid_level_1']):
        if len(group) < 2:
            continue

        # Sample representative item
        sample = group.sample(1).iloc[0]
        prefix = f"<|sid_start|><|sid_{sid_0}|><|sid_{sid_1 + 256}|>"

        conversations.append(create_conversation(
            f"What category do products with semantic ID prefix {prefix} belong to?",
            sample['categories_text'],
            "type_d_prefix_category"
        ))

    # D2: Prefix to examples
    for prefix, group in items_df.groupby('sid_prefix_2'):
        if len(group) < 3:
            continue

        # Get prefix format
        first_item = group.iloc[0]
        sid_0 = first_item['semantic_id_0']
        sid_1 = first_item['semantic_id_1']
        prefix_formatted = f"<|sid_start|><|sid_{sid_0}|><|sid_{sid_1 + 256}|>"

        # Sample 3 examples
        examples = group.sample(min(3, len(group)))
        example_sids = ' '.join(examples['semantic_id'].tolist())

        conversations.append(create_conversation(
            f"Give me some example products that share the semantic prefix {prefix_formatted}.",
            example_sids,
            "type_d_prefix_examples"
        ))

    # D3: Similar items (shared prefix)
    sampled_items = items_df.sample(min(5000, len(items_df)))
    for _, item in tqdm(sampled_items.iterrows(), total=len(sampled_items), desc="Type D similar"):
        sid = item['semantic_id']
        prefix = item['sid_prefix_2']

        # Find similar items (same prefix, different item)
        similar = items_df[
            (items_df['sid_prefix_2'] == prefix) &
            (items_df['semantic_id'] != sid)
        ]

        if len(similar) > 0:
            similar_sid = similar.sample(1).iloc[0]['semantic_id']
            conversations.append(create_conversation(
                f"What is a product similar to {sid}?",
                similar_sid,
                "type_d_similar_items"
            ))

    return conversations


def generate_type_e_data(sequences_df: pd.DataFrame, items_df: pd.DataFrame) -> List[Dict]:
    """
    Type E: Multi-hop reasoning (co-purchase and transition patterns)
    Largest category: ~67% of all training data

    Note: sequences_df should be the training split to avoid data leakage
    Expected: ~720,000 samples
    """
    conversations = []

    # Build co-purchase graph from training sequences only
    copurchase_pairs = []
    for _, row in sequences_df.iterrows():
        sid_seq = row['semantic_id_sequence']
        # Note: This is items[:-2] from training split, excluding the target
        for i in range(len(sid_seq) - 1):
            copurchase_pairs.append((sid_seq[i], sid_seq[i + 1]))

    # Count frequencies
    from collections import defaultdict, Counter
    forward_graph = defaultdict(list)
    backward_graph = defaultdict(list)

    for sid_a, sid_b in copurchase_pairs:
        forward_graph[sid_a].append(sid_b)
        backward_graph[sid_b].append(sid_a)

    # E1: Forward co-purchase
    for sid_a, targets in tqdm(forward_graph.items(), desc="Type E forward"):
        if len(targets) < 2:
            continue

        # Get most common targets
        target_counts = Counter(targets)
        for sid_b, count in target_counts.most_common(min(30, len(target_counts))):
            conversations.append(create_conversation(
                f"A user just purchased {sid_a}. What might they buy next?",
                sid_b,
                "type_e_copurchase_forward"
            ))

    # E2: Backward co-purchase
    for sid_b, sources in tqdm(backward_graph.items(), desc="Type E backward"):
        if len(sources) < 2:
            continue

        # Get most common sources
        source_counts = Counter(sources)
        for sid_a, count in source_counts.most_common(min(30, len(source_counts))):
            conversations.append(create_conversation(
                f"What do users often purchase before buying {sid_b}?",
                sid_a,
                "type_e_copurchase_backward"
            ))

    # E3: Category transitions
    # Create category mapping
    sid_to_category = dict(zip(items_df['semantic_id'], items_df['categories_text']))

    category_transitions = []
    for sid_a, sid_b in copurchase_pairs:
        cat_a = sid_to_category.get(sid_a, "Unknown")
        cat_b = sid_to_category.get(sid_b, "Unknown")
        if cat_a != "Unknown" and cat_b != "Unknown":
            category_transitions.append((cat_a, cat_b, sid_b))

    # Sample transitions
    sampled_transitions = random.sample(
        category_transitions,
        min(100000, len(category_transitions))
    )

    for cat_a, cat_b, sid_b in tqdm(sampled_transitions, desc="Type E category"):
        conversations.append(create_conversation(
            f"After purchasing items from '{cat_a}', what product from '{cat_b}' might a user buy?",
            sid_b,
            "type_e_category_transition"
        ))

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Generate OneRec-Think LLM training data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing preprocessed parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save training data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit-type-c",
        type=int,
        default=None,
        help="Limit Type C samples for faster testing"
    )
    parser.add_argument(
        "--limit-type-e",
        type=int,
        default=None,
        help="Limit Type E samples for faster testing"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OneRec-Think LLM Training Data Generation")
    print("=" * 80)
    print("\nNote: Using leave-last-out protocol aligned with OneRec-Think original")
    print("  - Training data from items[:-2] predicting item[-2]")
    print("  - Validation set: Beauty_sequences_val.parquet (items[:-1] → item[-1])")
    print("  - Test set: Beauty_sequences_test.parquet (items[:] → next item)")

    # Load preprocessed data
    print(f"\nLoading preprocessed data from {data_dir}...")
    items_df = pd.read_parquet(data_dir / "Beauty_items_onerec.parquet")
    train_sequences_df = pd.read_parquet(data_dir / "Beauty_sequences_train.parquet")

    print(f"  Items: {len(items_df):,}")
    print(f"  Training sequences: {len(train_sequences_df):,}")

    # Generate all conversation types
    all_conversations = []

    print("\nGenerating Type A: Semantic ID → Text...")
    type_a = generate_type_a_data(items_df)
    print(f"  Generated {len(type_a):,} samples")
    all_conversations.extend(type_a)

    print("\nGenerating Type B: Text → Semantic ID...")
    type_b = generate_type_b_data(items_df)
    print(f"  Generated {len(type_b):,} samples")
    all_conversations.extend(type_b)

    print("\nGenerating Type C: Sequential Prediction...")
    type_c = generate_type_c_data(train_sequences_df)
    if args.limit_type_c and len(type_c) > args.limit_type_c:
        type_c = random.sample(type_c, args.limit_type_c)
        print(f"  Limited to {len(type_c):,} samples")
    else:
        print(f"  Generated {len(type_c):,} samples")
    all_conversations.extend(type_c)

    print("\nGenerating Type D: Semantic Understanding...")
    type_d = generate_type_d_data(items_df)
    print(f"  Generated {len(type_d):,} samples")
    all_conversations.extend(type_d)

    print("\nGenerating Type E: Multi-hop Reasoning...")
    type_e = generate_type_e_data(train_sequences_df, items_df)
    if args.limit_type_e and len(type_e) > args.limit_type_e:
        type_e = random.sample(type_e, args.limit_type_e)
        print(f"  Limited to {len(type_e):,} samples")
    else:
        print(f"  Generated {len(type_e):,} samples")
    all_conversations.extend(type_e)

    print(f"\nTotal training conversations: {len(all_conversations):,}")

    # Shuffle training data
    print("\nShuffling training data...")
    random.shuffle(all_conversations)
    train_data = all_conversations

    # Save training data
    print("\nSaving output file...")
    train_path = output_dir / "Beauty_conversations_train_onerec.parquet"

    pd.DataFrame(train_data).to_parquet(train_path, index=False)

    print(f"  Training data: {train_path} ({len(train_data):,} samples)")

    # Print statistics
    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)

    print("\nType Distribution (Training Set):")
    train_df = pd.DataFrame(train_data)
    type_counts = train_df['type'].value_counts()
    for type_name, count in type_counts.items():
        percentage = (count / len(train_data)) * 100
        print(f"  {type_name}: {count:,} ({percentage:.1f}%)")

    print("\nSample conversation:")
    sample = train_data[0]
    print(json.dumps(sample, indent=2))

    print("\n" + "=" * 80)
    print("Data Split Summary:")
    print("=" * 80)
    print(f"  Training conversations: {train_path}")
    print(f"    → {len(train_data):,} samples generated from items[:-2] predicting item[-2]")
    print(f"\n  Validation sequences: {data_dir / 'Beauty_sequences_val.parquet'}")
    print(f"    → Use for validation during training (items[:-1] → item[-1])")
    print(f"\n  Test sequences: {data_dir / 'Beauty_sequences_test.parquet'}")
    print(f"    → Use for final evaluation (items[:] → next item)")
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Validate data quality: python src/validate_training_data.py")
    print("  2. Start training: python src/finetune_qwen3_8b_vocab.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
