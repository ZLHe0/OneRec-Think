"""
Data preprocessing pipeline for OneRec-Think Beauty dataset.
Converts OneRec-Think format to semantic-ids-llm format.

This script:
1. Loads Beauty.pretrain.json with item metadata and semantic IDs
2. Loads sequential_data_processed.txt with user sequences
3. Parses and converts semantic IDs from OneRec format to semantic-ids-llm format
4. Creates items and sequences dataframes in parquet format

Expected output:
- data/Beauty_items_onerec.parquet: Item metadata with converted semantic IDs
- data/Beauty_sequences_onerec.parquet: User sequences with semantic ID mappings
"""

import json
import re
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple


def parse_onerec_sid(sid_string: str) -> List[int]:
    """
    Parse OneRec-Think semantic ID format.

    Input: <|sid_begin|><s_a_99><s_b_19><s_c_220><s_d_204><|sid_end|>
    Output: [99, 19, 220, 204]

    Args:
        sid_string: Semantic ID string in OneRec format

    Returns:
        List of 4 integers representing the hierarchical codes
    """
    matches = re.findall(r'<s_[a-d]_(\d+)>', sid_string)
    if len(matches) != 4:
        raise ValueError(f"Expected 4 semantic ID codes, got {len(matches)}: {sid_string}")
    return [int(m) for m in matches]


def convert_to_semantic_ids_llm_format(codes: List[int]) -> str:
    """
    Convert semantic ID codes to semantic-ids-llm format with offset encoding.

    Input: [99, 19, 220, 204]
    Output: <|sid_start|><|sid_99|><|sid_275|><|sid_732|><|sid_972|><|sid_end|>

    Offset encoding:
    - Level 0 (a): 0-255 → sid_0 to sid_255
    - Level 1 (b): 0-255 → sid_256 to sid_511
    - Level 2 (c): 0-255 → sid_512 to sid_767
    - Level 3 (d): 0-255 → sid_768 to sid_1023

    Args:
        codes: List of 4 integers [0-255] for each level

    Returns:
        Formatted semantic ID string in semantic-ids-llm format
    """
    if len(codes) != 4:
        raise ValueError(f"Expected 4 codes, got {len(codes)}")

    for i, code in enumerate(codes):
        if not 0 <= code <= 255:
            raise ValueError(f"Code at level {i} is {code}, expected 0-255")

    offsets = [0, 256, 512, 768]
    tokens = ["<|sid_start|>"]
    for level, code in enumerate(codes):
        tokens.append(f"<|sid_{offsets[level] + code}|>")
    tokens.append("<|sid_end|>")
    return "".join(tokens)


def load_items_data(items_path: Path, limit: int = None) -> pd.DataFrame:
    """
    Load item metadata from Beauty.pretrain.json and convert to dataframe.

    Args:
        items_path: Path to Beauty.pretrain.json
        limit: Optional limit on number of items to load (for testing)

    Returns:
        Pandas DataFrame with columns:
        - parent_asin: Item ID (string)
        - title: Product title
        - description_text: Product description
        - categories_text: Category path
        - item_context: Concatenated context
        - semantic_id_0/1/2/3: Individual semantic codes
        - semantic_id: Full semantic ID in semantic-ids-llm format
    """
    print(f"Loading items from {items_path}...")

    with open(items_path) as f:
        items_dict = json.load(f)

    print(f"Loaded {len(items_dict)} items")

    items_data = []
    parse_errors = 0

    for item_id, item_info in items_dict.items():
        if limit and len(items_data) >= limit:
            break

        try:
            # Parse semantic ID
            sid_codes = parse_onerec_sid(item_info["sid"])

            # Create item_context (concatenated metadata)
            item_context = f"{item_info['title']} | {item_info['description']} | {item_info['categories']}"

            items_data.append({
                "parent_asin": item_id,
                "title": item_info["title"],
                "description_text": item_info["description"],
                "categories_text": item_info["categories"],
                "item_context": item_context,
                "semantic_id_0": sid_codes[0],
                "semantic_id_1": sid_codes[1],
                "semantic_id_2": sid_codes[2],
                "semantic_id_3": sid_codes[3],
                "semantic_id": convert_to_semantic_ids_llm_format(sid_codes),
            })
        except (ValueError, KeyError) as e:
            parse_errors += 1
            print(f"Warning: Failed to parse item {item_id}: {e}")
            continue

    if parse_errors > 0:
        print(f"Warning: {parse_errors} items failed to parse")

    print(f"Successfully processed {len(items_data)} items")

    df = pd.DataFrame(items_data)

    # Print statistics
    print("\nItems DataFrame Statistics:")
    print(f"  Total items: {len(df)}")
    print(f"  Unique semantic IDs: {df['semantic_id'].nunique()}")
    print(f"  Average title length: {df['title'].str.len().mean():.1f} chars")
    print(f"  Average description length: {df['description_text'].str.len().mean():.1f} chars")

    return df


def create_train_val_test_splits(
    sequences_path: Path,
    id_to_sid: Dict[str, str],
    limit: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sequences and create train/val/test splits following OneRec-Think protocol.

    Strategy (leave-last-out):
    - Train: items[:-2] - use first (N-2) items, target is item[-2]
    - Val:   items[:-1] - use first (N-1) items, target is item[-1]
    - Test:  items[:]   - use all N items, target is next item (predicted during eval)

    Args:
        sequences_path: Path to sequential_data_processed.txt
        id_to_sid: Dictionary mapping item_id -> semantic_id
        limit: Optional limit on number of sequences to load (for testing)

    Returns:
        Tuple of (train_df, val_df, test_df)
        Each DataFrame has columns:
        - user_id, sequence, semantic_id_sequence, sequence_length, target_sid
    """
    print(f"\nLoading sequences and creating train/val/test splits from {sequences_path}...")

    train_data = []
    val_data = []
    test_data = []

    skipped_sequences = 0
    missing_items_count = 0

    with open(sequences_path) as f:
        for line_num, line in enumerate(f, 1):
            if limit and line_num > limit:
                break

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            user_id = parts[0]
            item_ids = parts[1:]

            # Map item IDs to semantic IDs
            semantic_id_seq = []
            valid_items = []

            for item_id in item_ids:
                if item_id in id_to_sid:
                    valid_items.append(item_id)
                    semantic_id_seq.append(id_to_sid[item_id])
                else:
                    missing_items_count += 1

            # Need at least 3 items for train (items[:-2] must have at least 1 item)
            if len(valid_items) < 3:
                skipped_sequences += 1
                continue

            # Train: use items[:-2] to predict item[-2]
            train_seq = semantic_id_seq[:-2]
            train_target = semantic_id_seq[-2]
            train_data.append({
                'user_id': user_id,
                'sequence': valid_items[:-2],
                'semantic_id_sequence': train_seq,
                'sequence_length': len(train_seq),
                'target_sid': train_target
            })

            # Val: use items[:-1] to predict item[-1]
            val_seq = semantic_id_seq[:-1]
            val_target = semantic_id_seq[-1]
            val_data.append({
                'user_id': user_id,
                'sequence': valid_items[:-1],
                'semantic_id_sequence': val_seq,
                'sequence_length': len(val_seq),
                'target_sid': val_target
            })

            # Test: use all items, target is "next" item (for real-world evaluation)
            test_data.append({
                'user_id': user_id,
                'sequence': valid_items,
                'semantic_id_sequence': semantic_id_seq,
                'sequence_length': len(valid_items),
                'target_sid': None  # Will be predicted in real evaluation
            })

    print(f"Processed {line_num} sequences from file")
    print(f"Skipped {skipped_sequences} sequences with < 3 items")
    print(f"Missing {missing_items_count} item mappings")

    print("\nCreated splits:")
    print(f"  Train: {len(train_data)} sequences")
    print(f"  Val:   {len(val_data)} sequences")
    print(f"  Test:  {len(test_data)} sequences")

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Print statistics for each split
    print("\nTrain split statistics:")
    print(f"  Total sequences: {len(train_df)}")
    print(f"  Unique users: {train_df['user_id'].nunique()}")
    print(f"  Average sequence length: {train_df['sequence_length'].mean():.1f}")
    print(f"  Min/Max length: {train_df['sequence_length'].min()}/{train_df['sequence_length'].max()}")

    print("\nVal split statistics:")
    print(f"  Total sequences: {len(val_df)}")
    print(f"  Average sequence length: {val_df['sequence_length'].mean():.1f}")
    print(f"  Min/Max length: {val_df['sequence_length'].min()}/{val_df['sequence_length'].max()}")

    print("\nTest split statistics:")
    print(f"  Total sequences: {len(test_df)}")
    print(f"  Average sequence length: {test_df['sequence_length'].mean():.1f}")
    print(f"  Min/Max length: {test_df['sequence_length'].min()}/{test_df['sequence_length'].max()}")

    return train_df, val_df, test_df


def main():
    """Main preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OneRec-Think data for semantic-ids-llm")
    parser.add_argument("--data-dir", type=str,
                       default="data",
                       help="Directory containing input data files")
    parser.add_argument("--output-dir", type=str,
                       default="data",
                       help="Directory to save output parquet files")
    parser.add_argument("--limit", type=int,
                       default=None,
                       help="Limit number of items/sequences to process (for testing)")
    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    items_path = data_dir / "Beauty.pretrain.json"
    sequences_path = data_dir / "sequential_data_processed.txt"

    # Validate input files exist
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")

    print("="*80)
    print("OneRec-Think Data Preprocessing Pipeline")
    print("="*80)

    # Step 1: Load and process items
    items_df = load_items_data(items_path, limit=args.limit)

    # Step 2: Create item_id -> semantic_id mapping
    print("\nCreating item ID to semantic ID mapping...")
    id_to_sid = dict(zip(items_df['parent_asin'], items_df['semantic_id']))
    print(f"Created mapping for {len(id_to_sid)} items")

    # Step 3: Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_splits(sequences_path, id_to_sid, limit=args.limit)

    # Step 4: Save output files
    items_output = output_dir / "Beauty_items_onerec.parquet"
    train_output = output_dir / "Beauty_sequences_train.parquet"
    val_output = output_dir / "Beauty_sequences_val.parquet"
    test_output = output_dir / "Beauty_sequences_test.parquet"

    print(f"\nSaving output files...")
    items_df.to_parquet(items_output, index=False)
    print(f"  Items: {items_output}")

    train_df.to_parquet(train_output, index=False)
    print(f"  Train sequences: {train_output}")

    val_df.to_parquet(val_output, index=False)
    print(f"  Val sequences: {val_output}")

    test_df.to_parquet(test_output, index=False)
    print(f"  Test sequences: {test_output}")

    print("\n" + "="*80)
    print("Preprocessing complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {items_output} ({len(items_df)} items)")
    print(f"  - {train_output} ({len(train_df)} sequences)")
    print(f"  - {val_output} ({len(val_df)} sequences)")
    print(f"  - {test_output} ({len(test_df)} sequences)")
    print(f"\nNext step: Run src/generate_training_data_onerec.py to generate LLM training data from train split")


if __name__ == "__main__":
    main()
