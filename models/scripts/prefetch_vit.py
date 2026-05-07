from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and cache the ViT-Large model used by the pipeline.")
    parser.add_argument(
        "--model-name",
        default="google/vit-large-patch16-224-in21k",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional local cache directory for Hugging Face assets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("prefetch_vit.log")

    from transformers import AutoImageProcessor, ViTModel

    cache_dir = str(Path(args.cache_dir).resolve()) if args.cache_dir else None

    start = time.time()
    logger.info("Downloading processor for %s", args.model_name)
    AutoImageProcessor.from_pretrained(args.model_name, cache_dir=cache_dir)
    logger.info("Processor cached in %.2f seconds", time.time() - start)

    model_start = time.time()
    logger.info("Downloading model weights for %s", args.model_name)
    ViTModel.from_pretrained(args.model_name, cache_dir=cache_dir)
    logger.info("Model cached in %.2f seconds", time.time() - model_start)


if __name__ == "__main__":
    main()
