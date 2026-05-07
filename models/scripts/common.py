from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODELS_ROOT = ARTIFACTS_ROOT / "models"
ARTIFACT_REPORTS_ROOT = ARTIFACTS_ROOT / "reports"
ARTIFACT_PLOTS_ROOT = ARTIFACTS_ROOT / "plots"
ARTIFACT_TABLES_ROOT = ARTIFACTS_ROOT / "tables"

DATASET_LABELS = ("NIH", "TCIA", "RSNA")
SUPPORTED_TABLE_SUFFIXES = (".xlsx", ".csv")

_DATASET_FILE_INDEX_CACHE: Dict[tuple[str, str], Dict[str, Path]] = {}


@dataclass(frozen=True)
class DatasetArtifacts:
    name: str
    slug: str
    bronze_path: Path
    silver_path: Path
    gold_path: Path
    summary_report_path: Path
    feature_report_path: Path


@dataclass
class PipelineConfig:
    raw_metadata_path: Path = PROJECT_ROOT / "raw_image_metadata.csv"
    image_root: Optional[Path] = None
    model_name: str = "google/vit-large-patch16-224-in21k"
    batch_size: int = 8
    feature_prefix: str = "feature_"
    max_missing_ratio: float = 0.40
    low_variance_threshold: float = 1e-8
    correlation_threshold: float = 0.98
    importance_threshold: float = 0.10
    label_column: str = "label"
    random_state: int = 42
    model_cache_dir: Optional[Path] = None
    max_rows: Optional[int] = None
    device: Optional[str] = None


def get_dataset_artifacts() -> Dict[str, DatasetArtifacts]:
    return {
        "NIH": DatasetArtifacts(
            name="NIH",
            slug="nih",
            bronze_path=PROJECT_ROOT / "bronze" / "nih" / "nih_metadata.csv",
            silver_path=PROJECT_ROOT / "silver" / "silver_nih.xlsx",
            gold_path=PROJECT_ROOT / "gold" / "gold_nih.xlsx",
            summary_report_path=PROJECT_ROOT / "reports" / "feature_selection_nih_summary.json",
            feature_report_path=PROJECT_ROOT / "reports" / "feature_selection_nih_scores.csv",
        ),
        "TCIA": DatasetArtifacts(
            name="TCIA",
            slug="tcia",
            bronze_path=PROJECT_ROOT / "bronze" / "tcia" / "tcia_metadata.csv",
            silver_path=PROJECT_ROOT / "silver" / "silver_tcia.xlsx",
            gold_path=PROJECT_ROOT / "gold" / "gold_tcia.xlsx",
            summary_report_path=PROJECT_ROOT / "reports" / "feature_selection_tcia_summary.json",
            feature_report_path=PROJECT_ROOT / "reports" / "feature_selection_tcia_scores.csv",
        ),
        "RSNA": DatasetArtifacts(
            name="RSNA",
            slug="rsna",
            bronze_path=PROJECT_ROOT / "bronze" / "rsna" / "rsna_metadata.csv",
            silver_path=PROJECT_ROOT / "silver" / "silver_rsna.xlsx",
            gold_path=PROJECT_ROOT / "gold" / "gold_rsna.xlsx",
            summary_report_path=PROJECT_ROOT / "reports" / "feature_selection_rsna_summary.json",
            feature_report_path=PROJECT_ROOT / "reports" / "feature_selection_rsna_scores.csv",
        ),
    }


def ensure_directories() -> None:
    for relative_path in (
        "bronze/nih",
        "bronze/tcia",
        "bronze/rsna",
        "silver",
        "gold",
        "reports",
        "logs",
        "warehouse",
        "artifacts/models",
        "artifacts/reports",
        "artifacts/plots",
        "artifacts/plots/eda",
        "artifacts/plots/feature_analysis",
        "artifacts/plots/training",
        "artifacts/plots/benchmarks",
        "artifacts/plots/confusion_matrices",
        "artifacts/plots/binary",
        "artifacts/plots/evaluation",
        "artifacts/tables",
    ):
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def setup_logging(log_name: str) -> logging.Logger:
    ensure_directories()
    log_path = PROJECT_ROOT / "logs" / log_name
    logger_name = f"image_pipeline.{Path(log_name).stem}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def normalize_dataset_name(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in DATASET_LABELS:
        return text
    return None


def normalize_label_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return text


def canonicalize_label_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).map(normalize_label_text)
    if normalized.empty:
        return normalized

    canonical_map: Dict[str, str] = {}
    temp = pd.DataFrame({"original": normalized, "key": normalized.str.casefold()})
    for key, group in temp.groupby("key", sort=False):
        canonical_map[key] = group["original"].value_counts().idxmax()
    return temp["key"].map(canonical_map)


def normalize_split_value(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "testing": "test",
        "holdout": "test",
    }
    return mapping.get(text)


def resolve_optional_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def select_dataset_artifacts(requested_datasets: Optional[Iterable[str]] = None) -> List[DatasetArtifacts]:
    artifacts = get_dataset_artifacts()
    if not requested_datasets:
        return [artifacts[label] for label in DATASET_LABELS]

    selected: List[DatasetArtifacts] = []
    for item in requested_datasets:
        normalized = normalize_dataset_name(item)
        if normalized is None:
            raise ValueError(f"Unsupported dataset '{item}'. Expected one of: {', '.join(DATASET_LABELS)}")
        selected.append(artifacts[normalized])
    return selected


def resolve_existing_table_path(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path

    basename = preferred_path.stem
    parent = preferred_path.parent
    for suffix in SUPPORTED_TABLE_SUFFIXES:
        candidate = parent / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No table file found for {preferred_path}")


def read_dataframe(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    resolved_path = resolve_existing_table_path(path)
    if resolved_path.suffix.lower() == ".csv":
        return pd.read_csv(resolved_path, nrows=nrows)
    if resolved_path.suffix.lower() == ".xlsx":
        return pd.read_excel(resolved_path, engine="openpyxl", nrows=nrows)
    raise ValueError(f"Unsupported table format: {resolved_path.suffix}")


def write_json(path: Path, payload: dict) -> None:
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def find_feature_columns(columns: Iterable[str], prefix: str = "feature_") -> List[str]:
    return [column for column in columns if str(column).startswith(prefix)]


def build_image_candidates(raw_path: str, dataset_name: str, image_root: Optional[Path]) -> List[Path]:
    path_value = Path(str(raw_path)).expanduser()
    candidates: List[Path] = []
    path_parts = list(path_value.parts)
    trimmed_parts = path_parts[1:] if path_parts and path_parts[0].lower() == "images" else path_parts
    filename = path_value.name

    if path_value.is_absolute():
        candidates.append(path_value)
    else:
        candidates.append(PROJECT_ROOT / path_value)
        candidates.append(PROJECT_ROOT / "images" / path_value)

        if image_root is not None:
            candidates.append(image_root / path_value)
            if trimmed_parts:
                candidates.append(image_root / Path(*trimmed_parts))

            dataset_root = image_root / dataset_name
            candidates.append(dataset_root / filename)
            candidates.append(dataset_root / "images" / filename)
            candidates.append(dataset_root / "Images" / filename)
            candidates.append(dataset_root / "Training" / "Images" / filename)
            candidates.append(dataset_root / "Test" / "Images" / filename)
            candidates.append(image_root / filename)

    deduplicated: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(candidate)
    return deduplicated


def build_dataset_file_index(image_root: Optional[Path], dataset_name: str) -> Dict[str, Path]:
    if image_root is None:
        return {}

    cache_key = (str(image_root.resolve(strict=False)), dataset_name.upper())
    if cache_key in _DATASET_FILE_INDEX_CACHE:
        return _DATASET_FILE_INDEX_CACHE[cache_key]

    dataset_root = image_root / dataset_name
    if not dataset_root.exists():
        _DATASET_FILE_INDEX_CACHE[cache_key] = {}
        return {}

    file_index: Dict[str, Path] = {}
    for candidate in dataset_root.rglob("*"):
        if candidate.is_file():
            file_index.setdefault(candidate.name, candidate)

    _DATASET_FILE_INDEX_CACHE[cache_key] = file_index
    return file_index


def resolve_image_path(raw_path: str, dataset_name: str, image_root: Optional[Path]) -> Optional[Path]:
    for candidate in build_image_candidates(raw_path=raw_path, dataset_name=dataset_name, image_root=image_root):
        if candidate.exists():
            return candidate
    dataset_file_index = build_dataset_file_index(image_root=image_root, dataset_name=dataset_name)
    return dataset_file_index.get(Path(str(raw_path)).name)
