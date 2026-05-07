from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    DATASET_LABELS,
    PipelineConfig,
    ensure_directories,
    normalize_dataset_name,
    select_dataset_artifacts,
    setup_logging,
)


def infer_dataset_from_path(path_value: object) -> str | None:
    if path_value is None or pd.isna(path_value):
        return None

    normalized_path = str(path_value).replace("\\", "/").upper()
    for dataset_name in DATASET_LABELS:
        token = f"/{dataset_name}/"
        if token in normalized_path or normalized_path.startswith(f"{dataset_name}/"):
            return dataset_name
    return None


def resolve_dataset_labels(df: pd.DataFrame) -> pd.Series:
    if "dataset" in df.columns:
        dataset_series = df["dataset"].map(normalize_dataset_name)
    else:
        dataset_series = pd.Series([None] * len(df), index=df.index, dtype="object")

    if "path" in df.columns:
        inferred_series = df["path"].map(infer_dataset_from_path)
        dataset_series = dataset_series.fillna(inferred_series)

    return dataset_series


def split_raw_metadata(
    config: PipelineConfig,
    requested_datasets: List[str] | None = None,
) -> Dict[str, int]:
    logger = setup_logging("data_splitter.log")
    ensure_directories()

    logger.info("Loading raw metadata from %s", config.raw_metadata_path)
    df = pd.read_csv(config.raw_metadata_path)
    logger.info("Loaded %s records with %s columns", len(df), len(df.columns))

    resolved_dataset = resolve_dataset_labels(df)
    unresolved_mask = resolved_dataset.isna()
    if unresolved_mask.any():
        sample_rows = df.loc[unresolved_mask, df.columns.intersection(["image_id", "dataset", "path"])].head(10)
        raise ValueError(
            "Unable to resolve dataset name for some records. "
            f"Examples:\n{sample_rows.to_string(index=False)}"
        )

    df = df.copy()
    df["dataset"] = resolved_dataset

    selected_artifacts = select_dataset_artifacts(requested_datasets)
    counts: Dict[str, int] = {}

    for artifacts in selected_artifacts:
        subset = df.loc[df["dataset"] == artifacts.name].copy()
        artifacts.bronze_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(artifacts.bronze_path, index=False)
        counts[artifacts.name] = int(len(subset))
        logger.info(
            "Saved %s rows for %s to %s",
            len(subset),
            artifacts.name,
            artifacts.bronze_path,
        )

    logger.info("Bronze split complete: %s", counts)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split raw image metadata into NIH, TCIA, and RSNA bronze files.")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(raw_metadata_path=Path(args.input).resolve())
    split_raw_metadata(config=config, requested_datasets=args.datasets)


if __name__ == "__main__":
    main()
