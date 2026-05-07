from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    PipelineConfig,
    ensure_directories,
    select_dataset_artifacts,
    setup_logging,
    write_json,
)


def build_feature_report_frame(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    report = pd.DataFrame(index=feature_columns)
    report.index.name = "feature"
    report["missing_ratio"] = numeric_features.isna().mean()
    report["non_null_count"] = numeric_features.notna().sum()
    report["nunique"] = numeric_features.nunique(dropna=True)
    report["variance"] = np.nan
    report["raw_score"] = np.nan
    report["normalized_score"] = np.nan
    report["selected"] = False
    report["removal_reason"] = "pending"
    report["correlated_with"] = ""
    return report


def median_impute(frame: pd.DataFrame) -> pd.DataFrame:
    medians = frame.median(axis=0)
    return frame.fillna(medians)


def prune_correlated_features(
    frame: pd.DataFrame,
    variance_by_feature: pd.Series,
    missing_ratio_by_feature: pd.Series,
    correlation_threshold: float,
) -> Tuple[List[str], Dict[str, str]]:
    if frame.shape[1] <= 1:
        return list(frame.columns), {}

    ordered_columns = sorted(
        frame.columns,
        key=lambda column: (-float(variance_by_feature[column]), float(missing_ratio_by_feature[column]), column),
    )
    corr_matrix = frame[ordered_columns].corr().abs()
    kept: List[str] = []
    dropped: Dict[str, str] = {}

    for column in ordered_columns:
        correlated_with = next(
            (
                kept_column
                for kept_column in kept
                if float(corr_matrix.loc[column, kept_column]) >= correlation_threshold
            ),
            None,
        )
        if correlated_with is None:
            kept.append(column)
        else:
            dropped[column] = correlated_with

    return kept, dropped


def compute_feature_scores(
    feature_frame: pd.DataFrame,
    target: Optional[pd.Series],
    random_state: int,
) -> Tuple[np.ndarray, str]:
    if feature_frame.shape[1] == 0:
        return np.array([], dtype=np.float64), "none"

    if target is not None and target.nunique(dropna=True) > 1:
        encoded_target = target.fillna("missing").astype(str).astype("category").cat.codes.to_numpy()
        raw_scores = mutual_info_classif(
            feature_frame.to_numpy(dtype=np.float32),
            encoded_target,
            random_state=random_state,
        )
        return np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0), "mutual_info_classif"

    raw_scores = feature_frame.var(axis=0, ddof=0).to_numpy(dtype=np.float64)
    return np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0), "variance"


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    if raw_scores.size == 0:
        return raw_scores
    max_score = float(np.max(raw_scores))
    if max_score <= 0:
        return np.zeros_like(raw_scores, dtype=np.float64)
    return raw_scores / max_score


def select_features_for_dataset(
    input_path: Path,
    output_path: Path,
    summary_report_path: Path,
    feature_report_path: Path,
    config: PipelineConfig,
) -> Dict[str, object]:
    logger = setup_logging("feature_selector.log")
    ensure_directories()

    logger.info("Loading silver dataset from %s", input_path)
    df = pd.read_excel(input_path, engine="openpyxl")
    feature_columns = [column for column in df.columns if column.startswith(config.feature_prefix)]
    if not feature_columns:
        raise ValueError(f"No feature columns starting with '{config.feature_prefix}' were found in {input_path}")

    report = build_feature_report_frame(df, feature_columns)
    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    high_missing_columns = report.index[report["missing_ratio"] > config.max_missing_ratio].tolist()
    if high_missing_columns:
        report.loc[high_missing_columns, "removal_reason"] = "high_missingness"

    remaining_columns = [column for column in feature_columns if column not in high_missing_columns]
    remaining_frame = numeric_features[remaining_columns]

    nunique = remaining_frame.nunique(dropna=True)
    constant_columns = nunique[nunique <= 1].index.tolist()
    if constant_columns:
        report.loc[constant_columns, "removal_reason"] = "constant"

    variance_input_columns = [column for column in remaining_columns if column not in constant_columns]
    variance_frame = median_impute(remaining_frame[variance_input_columns])
    variances = variance_frame.var(axis=0, ddof=0)
    report.loc[variances.index, "variance"] = variances.astype(float)

    low_variance_columns = variances[variances < config.low_variance_threshold].index.tolist()
    if low_variance_columns:
        report.loc[low_variance_columns, "removal_reason"] = "low_variance"

    correlation_input_columns = [column for column in variance_input_columns if column not in low_variance_columns]
    correlation_frame = variance_frame[correlation_input_columns]
    kept_after_correlation, correlated_drops = prune_correlated_features(
        frame=correlation_frame,
        variance_by_feature=variances,
        missing_ratio_by_feature=report["missing_ratio"],
        correlation_threshold=config.correlation_threshold,
    )
    for dropped_column, kept_column in correlated_drops.items():
        report.loc[dropped_column, "removal_reason"] = "high_correlation"
        report.loc[dropped_column, "correlated_with"] = kept_column

    scoring_frame = correlation_frame[kept_after_correlation]
    target = df[config.label_column] if config.label_column in df.columns else None
    raw_scores, scoring_method = compute_feature_scores(
        feature_frame=scoring_frame,
        target=target,
        random_state=config.random_state,
    )
    normalized_scores = normalize_scores(raw_scores)

    if len(kept_after_correlation) > 0:
        report.loc[kept_after_correlation, "raw_score"] = raw_scores.astype(float)
        report.loc[kept_after_correlation, "normalized_score"] = normalized_scores.astype(float)

    selected_feature_set = {
        column
        for column, score in zip(kept_after_correlation, normalized_scores)
        if float(score) >= config.importance_threshold
    }
    below_threshold = [column for column in kept_after_correlation if column not in selected_feature_set]
    if below_threshold:
        report.loc[below_threshold, "removal_reason"] = "below_importance_threshold"

    if selected_feature_set:
        report.loc[list(selected_feature_set), "selected"] = True
        report.loc[list(selected_feature_set), "removal_reason"] = "selected"

    selected_feature_columns = [column for column in feature_columns if column in selected_feature_set]
    metadata_columns = [column for column in df.columns if not column.startswith(config.feature_prefix)]
    gold_df = df[metadata_columns + selected_feature_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_excel(output_path, index=False, engine="openpyxl")

    ordered_report = report.reset_index().sort_values(
        by=["selected", "normalized_score", "variance", "feature"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    feature_report_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_report.to_csv(feature_report_path, index=False)

    removal_counts = (
        ordered_report.loc[~ordered_report["selected"], "removal_reason"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary = {
        "dataset": str(df["dataset"].iloc[0]) if "dataset" in df.columns and not df.empty else input_path.stem,
        "input_rows": int(len(df)),
        "input_features": int(len(feature_columns)),
        "selected_features": int(len(selected_feature_columns)),
        "scoring_method": scoring_method,
        "label_column": config.label_column if config.label_column in df.columns else None,
        "label_classes": int(target.nunique(dropna=True)) if target is not None else 0,
        "thresholds": {
            "max_missing_ratio": float(config.max_missing_ratio),
            "low_variance_threshold": float(config.low_variance_threshold),
            "correlation_threshold": float(config.correlation_threshold),
            "importance_threshold": float(config.importance_threshold),
        },
        "removed_counts": {key: int(value) for key, value in removal_counts.items()},
        "top_selected_features": ordered_report.loc[ordered_report["selected"], ["feature", "normalized_score"]]
        .head(15)
        .assign(normalized_score=lambda frame: frame["normalized_score"].fillna(0.0).astype(float))
        .to_dict(orient="records"),
        "gold_output_path": str(output_path),
        "feature_report_path": str(feature_report_path),
    }
    write_json(summary_report_path, summary)

    logger.info(
        "Selected %s/%s features for %s using %s",
        len(selected_feature_columns),
        len(feature_columns),
        input_path.name,
        scoring_method,
    )
    return summary


def run_feature_selection(
    config: PipelineConfig,
    requested_datasets: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for artifacts in select_dataset_artifacts(requested_datasets):
        results[artifacts.name] = select_features_for_dataset(
            input_path=artifacts.silver_path,
            output_path=artifacts.gold_path,
            summary_report_path=artifacts.summary_report_path,
            feature_report_path=artifacts.feature_report_path,
            config=config,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select important image features and write gold outputs.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of datasets to process. Supported values: NIH TCIA RSNA.",
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
        help="Random seed for mutual information scoring.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        importance_threshold=args.importance_threshold,
        max_missing_ratio=args.max_missing_ratio,
        low_variance_threshold=args.low_variance_threshold,
        correlation_threshold=args.correlation_threshold,
        label_column=args.label_column,
        random_state=args.random_state,
    )
    run_feature_selection(config=config, requested_datasets=args.datasets)


if __name__ == "__main__":
    main()
