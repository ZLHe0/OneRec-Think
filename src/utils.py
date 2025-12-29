#!/usr/bin/env python3
"""
Utilities for OneRec-Think LLM training
Includes: logging, device management, and test prompts
"""

import logging
import sys
import torch
from pathlib import Path
from datetime import datetime


# ============================================================================
# Logging
# ============================================================================

def setup_logger(name: str, log_to_file: bool = True, log_dir: Path = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_to_file: Whether to log to a file
        log_dir: Directory for log files (default: logs/)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


# ============================================================================
# Device Management
# ============================================================================

class DeviceManager:
    """Manages device selection and provides device information."""

    def __init__(self, logger: logging.Logger = None):
        """Initialize device manager."""
        self.logger = logger
        self.device = self._select_device()
        self._log_device_info()

    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _log_device_info(self):
        """Log device information."""
        if self.logger is None:
            return

        self.logger.info("=== Device Information ===")
        self.logger.info(f"Selected device: {self.device}")

        if self.device.type == "cuda":
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")

            # Memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU memory: {total_memory:.2f} GB")

        elif self.device.type == "mps":
            self.logger.info("Using Apple Metal Performance Shaders (MPS)")

        else:
            self.logger.info("Using CPU (training will be slow)")

        self.logger.info("=" * 50)

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        stats = {}
        if self.device.type == "cuda":
            stats['allocated'] = torch.cuda.memory_allocated(0) / 1024**3
            stats['reserved'] = torch.cuda.memory_reserved(0) / 1024**3
            stats['max_reserved'] = torch.cuda.max_memory_reserved(0) / 1024**3
        return stats


# ============================================================================
# Test Prompts
# ============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_X|><|sid_Y|><|sid_Z|><|sid_W|><|sid_end|> that encode product relationships and categories."""


# Test prompts for Beauty domain
BEAUTY_TEST_PROMPTS = [
    # Type A: sid_to_text
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the title of the product with semantic ID <|sid_start|><|sid_99|><|sid_275|><|sid_732|><|sid_972|><|sid_end|>?"}
    ],

    # Type B: text_to_sid
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the semantic ID for a hair care product?"}
    ],

    # Type C: Sequential prediction
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "A user purchased these beauty items in sequence: <|sid_start|><|sid_50|><|sid_300|><|sid_600|><|sid_900|><|sid_end|> <|sid_start|><|sid_75|><|sid_350|><|sid_650|><|sid_850|><|sid_end|>. What might they buy next?"}
    ],

    # Type D: Semantic understanding
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What category do products with semantic ID prefix <|sid_start|><|sid_99|><|sid_275|> belong to?"}
    ],

    # Type E: Co-purchase
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "A user just purchased <|sid_start|><|sid_100|><|sid_300|><|sid_500|><|sid_800|><|sid_end|>. What might they buy next?"}
    ],

    # General recommendation
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Recommend a beauty product for skin care."}
    ],
]


# Alias for backward compatibility
REC_TEST_PROMPTS = BEAUTY_TEST_PROMPTS
