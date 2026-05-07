from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import PipelineConfig, ensure_directories, setup_logging  # noqa: E402
from scripts.data_splitter import split_raw_metadata  # noqa: E402
from scripts.feature_extractor import run_feature_extraction  # noqa: E402
from scripts.feature_selector import run_feature_selection  # noqa: E402


def run_pipeline(
    config: PipelineConfig,
    requested_datasets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    logger = setup_logging("pipeline.log")
    ensure_directories()

    logger.info("Starting bronze/silver/gold image pipeline")
    bronze_summary = split_raw_metadata(config=config, requested_datasets=list(requested_datasets) if requested_datasets else None)
    logger.info("Bronze layer complete: %s", bronze_summary)

    silver_summary = run_feature_extraction(
        config=config,
        requested_datasets=requested_datasets,
        dry_run=dry_run,
    )
    logger.info("Silver layer result: %s", silver_summary)

    if dry_run:
        logger.info("Dry run finished after metadata split and image path validation")
        return {
            "bronze": bronze_summary,
            "silver_validation": silver_summary,
        }

    gold_summary = run_feature_selection(
        config=config,
        requested_datasets=requested_datasets,
    )
    logger.info("Gold layer complete: %s", gold_summary)
    logger.info("Pipeline completed successfully")

    return {
        "bronze": bronze_summary,
        "silver": silver_summary,
        "gold": gold_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete bronze/silver/gold image metadata pipeline.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "raw_image_metadata.csv"),
        help="Path to the raw metadata CSV.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of datasets to process. Supported values: NIH TCIA RSNA.",
    )
    parser.add_argument(
        "--image-root",
        help="Optional image root directory when the metadata paths are not reachable from the project root.",
    )
    parser.add_argument(
        "--model-name",
        default="google/vit-large-patch16-224-in21k",
        help="Hugging Face model name for ViT-Large feature extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for feature extraction.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional local cache directory for Hugging Face assets.",
    )
    parser.add_argument(
        "--device",
        help="Torch device override, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.10,
        help="Keep features with normalized score greater than or equal to this threshold.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.40,
        help="Drop features above this missing-value ratio.",
    )
    parser.add_argument(
        "--low-variance-threshold",
        type=float,
        default=1e-8,
        help="Drop features with variance below this threshold after imputation.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.98,
        help="Drop features whose absolute correlation with an already kept feature exceeds this threshold.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Target column used for supervised feature scoring when more than one class is available.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for feature scoring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate splitting and image path resolution without loading the ViT model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        raw_metadata_path=Path(args.input).resolve(),
        image_root=Path(args.image_root).resolve() if args.image_root else None,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_missing_ratio=args.max_missing_ratio,
        low_variance_threshold=args.low_variance_threshold,
        correlation_threshold=args.correlation_threshold,
        importance_threshold=args.importance_threshold,
        label_column=args.label_column,
        random_state=args.random_state,
        model_cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
        max_rows=args.max_rows,
        device=args.device,
    )
    run_pipeline(config=config, requested_datasets=args.datasets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
