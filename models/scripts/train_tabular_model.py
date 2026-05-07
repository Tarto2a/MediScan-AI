from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    ARTIFACT_REPORTS_ROOT,
    MODELS_ROOT,
    canonicalize_label_series,
    ensure_directories,
    find_feature_columns,
    get_dataset_artifacts,
    normalize_dataset_name,
    normalize_split_value,
    read_dataframe,
    resolve_existing_table_path,
    resolve_image_path,
    setup_logging,
    write_json,
    write_text,
)


@dataclass
class TrainingConfig:
    image_root: Optional[Path]
    random_state: int
    test_size: float
    val_size: float
    feature_prefix: str
    feature_set_mode: str
    model_type: str
    search_iterations: int
    selection_metric: str
    max_rows_per_dataset: Optional[int]
    use_existing_split: bool
    feature_extractor_model_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a final tabular classifier on the gold-layer feature sets.")
    parser.add_argument("--image-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--feature-set-mode", choices=("union", "intersection"), default="union")
    parser.add_argument("--model-type", choices=("xgboost", "extratrees"), default="xgboost")
    parser.add_argument("--search-iterations", type=int, default=6)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--ignore-existing-split", action="store_true")
    parser.add_argument("--feature-extractor-model", default="google/vit-large-patch16-224-in21k")
    return parser


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    image_root = Path(args.image_root).resolve() if args.image_root else None
    return TrainingConfig(
        image_root=image_root,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        feature_prefix="feature_",
        feature_set_mode=args.feature_set_mode,
        model_type=args.model_type,
        search_iterations=max(1, args.search_iterations),
        selection_metric=args.selection_metric,
        max_rows_per_dataset=args.max_rows_per_dataset,
        use_existing_split=not args.ignore_existing_split,
        feature_extractor_model_name=args.feature_extractor_model,
    )


def default_gold_paths() -> List[Path]:
    dataset_artifacts = get_dataset_artifacts()
    return [resolve_existing_table_path(dataset_artifacts[name].gold_path) for name in ("NIH", "TCIA", "RSNA")]


def standardize_gold_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    source_path: Path,
    config: TrainingConfig,
    logger,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = dataframe.copy()
    rename_map = {}
    if "path" in df.columns and "image_path" not in df.columns:
        rename_map["path"] = "image_path"
    if "dataset" in df.columns and "dataset_name" not in df.columns:
        rename_map["dataset"] = "dataset_name"
    df.rename(columns=rename_map, inplace=True)

    if config.max_rows_per_dataset is not None:
        df = df.head(config.max_rows_per_dataset).copy()

    if "label" not in df.columns:
        raise ValueError(f"'label' column is required in {source_path}")

    df["label"] = canonicalize_label_series(df["label"])
    df = df.loc[df["label"].notna() & (df["label"].astype(str).str.len() > 0)].copy()

    if "dataset_name" not in df.columns:
        df["dataset_name"] = dataset_name
    df["dataset_name"] = df["dataset_name"].map(lambda value: normalize_dataset_name(value) or dataset_name)
    df["source_table"] = source_path.name

    if "split" in df.columns:
        df["split"] = df["split"].map(normalize_split_value)

    resolved_count = 0
    unresolved_count = 0
    if "image_path" in df.columns:
        resolved_paths: List[Optional[str]] = []
        for row in df[["image_path", "dataset_name"]].itertuples(index=False):
            resolved = resolve_image_path(raw_path=row.image_path, dataset_name=row.dataset_name, image_root=config.image_root)
            if resolved is not None:
                resolved_paths.append(str(resolved))
                resolved_count += 1
            else:
                resolved_paths.append(str(row.image_path))
                unresolved_count += 1
        df["image_path"] = resolved_paths
    else:
        df["image_path"] = None

    feature_columns = find_feature_columns(df.columns, prefix=config.feature_prefix)
    if not feature_columns:
        raise ValueError(f"No selected feature columns were found in {source_path}")

    summary = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "rows": int(len(df)),
        "feature_columns": int(len(feature_columns)),
        "resolved_image_paths": int(resolved_count),
        "unresolved_image_paths": int(unresolved_count),
        "label_distribution": {str(key): int(value) for key, value in df["label"].value_counts().sort_index().items()},
    }
    logger.info("Loaded %s with %s rows and %s selected features", source_path.name, len(df), len(feature_columns))
    return df, summary


def choose_feature_columns(frames: Sequence[pd.DataFrame], feature_prefix: str, mode: str) -> List[str]:
    feature_sets = [set(find_feature_columns(frame.columns, prefix=feature_prefix)) for frame in frames]
    if not feature_sets:
        return []
    selected = set.intersection(*feature_sets) if mode == "intersection" else set.union(*feature_sets)
    return sorted(selected)


def metadata_columns_in_order(frames: Sequence[pd.DataFrame], feature_columns: Sequence[str]) -> List[str]:
    feature_set = set(feature_columns)
    preferred_order = [
        "dataset_name",
        "label",
        "split",
        "image_path",
        "image_id",
        "patient_id",
        "bronze_row_number",
        "source_file_name",
        "ingestion_timestamp",
        "source_table",
    ]
    seen = set()
    ordered: List[str] = []
    for column in preferred_order:
        if any(column in frame.columns for frame in frames) and column not in feature_set and column not in seen:
            ordered.append(column)
            seen.add(column)
    for frame in frames:
        for column in frame.columns:
            if column not in feature_set and column not in seen:
                ordered.append(column)
                seen.add(column)
    return ordered


def combine_gold_dataframes(frames: Sequence[pd.DataFrame], feature_columns: Sequence[str]) -> pd.DataFrame:
    metadata_columns = metadata_columns_in_order(frames=frames, feature_columns=feature_columns)
    combined_frames = [frame.reindex(columns=metadata_columns + list(feature_columns)) for frame in frames]
    return pd.concat(combined_frames, ignore_index=True, sort=False)


def safe_stratified_split(
    indices: Sequence[int],
    labels: Sequence[int],
    test_size: float,
    random_state: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray, str]:
    index_array = np.asarray(indices)
    label_array = np.asarray(labels)
    try:
        train_idx, test_idx = train_test_split(
            index_array,
            test_size=test_size,
            stratify=label_array,
            random_state=random_state,
        )
        return train_idx, test_idx, "stratified"
    except ValueError as exc:
        logger.warning("Falling back to random split because stratification failed: %s", exc)
        train_idx, test_idx = train_test_split(
            index_array,
            test_size=test_size,
            random_state=random_state,
        )
        return train_idx, test_idx, "random_fallback"


def assign_splits(
    dataframe: pd.DataFrame,
    encoded_labels: np.ndarray,
    config: TrainingConfig,
    logger,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = dataframe.copy()
    split_info: Dict[str, object] = {
        "strategy": "generated",
        "source": "new_stratified_split",
        "details": {},
    }

    if config.use_existing_split and "split" in df.columns:
        normalized_split = df["split"].map(normalize_split_value)
        unique_splits = set(normalized_split.dropna().unique())
        if normalized_split.notna().all() and {"train", "test"}.issubset(unique_splits):
            df["split"] = normalized_split
            if "val" in unique_splits:
                split_info.update({"strategy": "existing", "source": "gold_split_column"})
                return df, split_info

            train_mask = df["split"] == "train"
            train_indices = df.index[train_mask].to_numpy()
            relative_val_size = min(max(config.val_size / max(1.0 - config.test_size, 1e-6), 0.05), 0.40)
            new_train_idx, new_val_idx, mode = safe_stratified_split(
                indices=train_indices,
                labels=encoded_labels[train_indices],
                test_size=relative_val_size,
                random_state=config.random_state,
                logger=logger,
            )
            df.loc[new_train_idx, "split"] = "train"
            df.loc[new_val_idx, "split"] = "val"
            split_info.update(
                {
                    "strategy": "existing_plus_validation",
                    "source": "gold_split_column_with_generated_val",
                    "details": {"train_val_split_mode": mode},
                }
            )
            return df, split_info

    indices = np.arange(len(df))
    train_val_idx, test_idx, test_mode = safe_stratified_split(
        indices=indices,
        labels=encoded_labels,
        test_size=config.test_size,
        random_state=config.random_state,
        logger=logger,
    )
    effective_val_ratio = min(max(config.val_size / max(1.0 - config.test_size, 1e-6), 0.05), 0.40)
    train_idx, val_idx, val_mode = safe_stratified_split(
        indices=train_val_idx,
        labels=encoded_labels[train_val_idx],
        test_size=effective_val_ratio,
        random_state=config.random_state,
        logger=logger,
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    split_info["details"] = {
        "train_test_split_mode": test_mode,
        "train_val_split_mode": val_mode,
    }
    return df, split_info


def compute_sample_weights_array(labels: np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=labels)


def xgboost_candidates(search_iterations: int) -> List[Dict[str, object]]:
    candidates = [
        {"n_estimators": 240, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 1.0, "min_child_weight": 1.0},
        {"n_estimators": 320, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.5, "min_child_weight": 1.0},
        {"n_estimators": 420, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 2.0, "min_child_weight": 2.0},
        {"n_estimators": 280, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.95, "colsample_bytree": 0.75, "reg_lambda": 1.0, "min_child_weight": 1.0},
        {"n_estimators": 360, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3.0, "min_child_weight": 2.0},
        {"n_estimators": 220, "max_depth": 8, "learning_rate": 0.06, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 2.5, "min_child_weight": 3.0},
    ]
    return candidates[: max(1, min(search_iterations, len(candidates)))]


def extratrees_candidates(search_iterations: int) -> List[Dict[str, object]]:
    candidates = [
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 30, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.8},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.7},
    ]
    return candidates[: max(1, min(search_iterations, len(candidates)))]


def build_model(model_type: str, params: Dict[str, object], num_classes: int, random_state: int):
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "xgboost is not installed. Add it to the environment or run with --model-type extratrees."
            ) from exc

        model_params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "reg_lambda": params["reg_lambda"],
            "min_child_weight": params["min_child_weight"],
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
            "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
            "missing": np.nan,
        }
        if num_classes > 2:
            model_params["num_class"] = num_classes
        return XGBClassifier(**model_params)

    if model_type == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def candidate_parameter_sets(model_type: str, search_iterations: int) -> List[Dict[str, object]]:
    if model_type == "xgboost":
        return xgboost_candidates(search_iterations)
    return extratrees_candidates(search_iterations)


def compute_roc_auc_metrics(
    y_true: np.ndarray,
    probabilities: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Optional[float]]:
    if probabilities is None:
        return {"roc_auc": None}

    try:
        if len(class_names) == 2:
            return {"roc_auc": float(roc_auc_score(y_true, probabilities[:, 1]))}

        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        if y_true_bin.shape[1] != len(class_names) or np.any(y_true_bin.sum(axis=0) == 0):
            return {"roc_auc_ovr_macro": None, "roc_auc_ovr_weighted": None}

        return {
            "roc_auc_ovr_macro": float(roc_auc_score(y_true_bin, probabilities, multi_class="ovr", average="macro")),
            "roc_auc_ovr_weighted": float(roc_auc_score(y_true_bin, probabilities, multi_class="ovr", average="weighted")),
        }
    except ValueError:
        if len(class_names) == 2:
            return {"roc_auc": None}
        return {"roc_auc_ovr_macro": None, "roc_auc_ovr_weighted": None}


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray]:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            row["label"]: {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1-score": float(row["f1-score"]),
                "support": int(row["support"]),
            }
            for _, row in report_df.iterrows()
            if row["label"] in class_names
        },
    }
    metrics.update(compute_roc_auc_metrics(y_true, probabilities, class_names))
    return metrics, report_df, cm


def selection_score(metrics: Dict[str, object], selection_metric: str) -> float:
    metric_map = {
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
    }
    return metric_map[selection_metric]


def save_confusion_matrix_image(matrix: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(class_names))
    display.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def training_report_paths() -> Dict[str, Path]:
    return {
        "model": MODELS_ROOT / "best_tabular_model.joblib",
        "class_names": MODELS_ROOT / "class_names.json",
        "selected_feature_columns": MODELS_ROOT / "selected_feature_columns.json",
        "inference_config": MODELS_ROOT / "inference_config.json",
        "training_report": ARTIFACT_REPORTS_ROOT / "training_report.json",
        "classification_report": ARTIFACT_REPORTS_ROOT / "classification_report.csv",
        "confusion_matrix": ARTIFACT_REPORTS_ROOT / "confusion_matrix.png",
        "metrics_summary": ARTIFACT_REPORTS_ROOT / "metrics_summary.txt",
    }


def run_training(config: TrainingConfig) -> Dict[str, object]:
    logger = setup_logging("train_tabular_model.log")
    ensure_directories()

    gold_frames: List[pd.DataFrame] = []
    gold_summaries: List[Dict[str, object]] = []
    for gold_path in default_gold_paths():
        dataset_name = normalize_dataset_name(gold_path.stem.split("_")[-1]) or gold_path.stem.split("_")[-1].upper()
        dataframe = read_dataframe(gold_path, nrows=config.max_rows_per_dataset)
        standardized_frame, summary = standardize_gold_dataframe(
            dataframe=dataframe,
            dataset_name=dataset_name,
            source_path=gold_path,
            config=config,
            logger=logger,
        )
        gold_frames.append(standardized_frame)
        gold_summaries.append(summary)

    feature_columns = choose_feature_columns(
        frames=gold_frames,
        feature_prefix=config.feature_prefix,
        mode=config.feature_set_mode,
    )
    if not feature_columns:
        raise RuntimeError("No gold-layer feature columns were available after alignment.")

    combined_df = combine_gold_dataframes(frames=gold_frames, feature_columns=feature_columns)
    combined_df["label"] = canonicalize_label_series(combined_df["label"])

    numeric_features = combined_df[feature_columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    valid_row_mask = combined_df["label"].notna() & numeric_features.notna().any(axis=1)
    dropped_rows = int((~valid_row_mask).sum())
    if dropped_rows:
        logger.warning("Dropping %s rows with missing labels or no usable feature values.", dropped_rows)
    combined_df = combined_df.loc[valid_row_mask].reset_index(drop=True)
    numeric_features = numeric_features.loc[valid_row_mask].reset_index(drop=True)

    class_names = sorted(combined_df["label"].astype(str).unique().tolist())
    label_to_index = {label: index for index, label in enumerate(class_names)}
    encoded_labels = combined_df["label"].map(label_to_index).to_numpy(dtype=np.int64)

    combined_df, split_info = assign_splits(dataframe=combined_df, encoded_labels=encoded_labels, config=config, logger=logger)

    split_masks = {
        "train": combined_df["split"] == "train",
        "val": combined_df["split"] == "val",
        "test": combined_df["split"] == "test",
    }
    if not all(mask.any() for mask in split_masks.values()):
        raise RuntimeError("Train/validation/test splits could not be constructed correctly.")

    X_train = numeric_features.loc[split_masks["train"], feature_columns]
    y_train = encoded_labels[split_masks["train"].to_numpy()]
    X_val = numeric_features.loc[split_masks["val"], feature_columns]
    y_val = encoded_labels[split_masks["val"].to_numpy()]
    X_test = numeric_features.loc[split_masks["test"], feature_columns]
    y_test = encoded_labels[split_masks["test"].to_numpy()]

    observed_train_classes = set(np.unique(y_train).tolist())
    expected_classes = set(range(len(class_names)))
    if observed_train_classes != expected_classes:
        logger.warning("Existing split omitted some classes from training. Regenerating a stratified split.")
        regenerated_config = TrainingConfig(
            image_root=config.image_root,
            random_state=config.random_state,
            test_size=config.test_size,
            val_size=config.val_size,
            feature_prefix=config.feature_prefix,
            feature_set_mode=config.feature_set_mode,
            model_type=config.model_type,
            search_iterations=config.search_iterations,
            selection_metric=config.selection_metric,
            max_rows_per_dataset=config.max_rows_per_dataset,
            use_existing_split=False,
            feature_extractor_model_name=config.feature_extractor_model_name,
        )
        combined_df, split_info = assign_splits(
            dataframe=combined_df,
            encoded_labels=encoded_labels,
            config=regenerated_config,
            logger=logger,
        )
        split_masks = {
            "train": combined_df["split"] == "train",
            "val": combined_df["split"] == "val",
            "test": combined_df["split"] == "test",
        }
        X_train = numeric_features.loc[split_masks["train"], feature_columns]
        y_train = encoded_labels[split_masks["train"].to_numpy()]
        X_val = numeric_features.loc[split_masks["val"], feature_columns]
        y_val = encoded_labels[split_masks["val"].to_numpy()]
        X_test = numeric_features.loc[split_masks["test"], feature_columns]
        y_test = encoded_labels[split_masks["test"].to_numpy()]

    class_weight_values = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weight_map = {class_names[index]: float(weight) for index, weight in enumerate(class_weight_values)}
    train_weights = compute_sample_weights_array(y_train)

    candidate_results: List[Dict[str, object]] = []
    best_model = None
    best_params: Dict[str, object] = {}
    best_val_metrics: Dict[str, object] = {}
    best_score = -math.inf

    for candidate_index, params in enumerate(candidate_parameter_sets(config.model_type, config.search_iterations), start=1):
        model = build_model(
            model_type=config.model_type,
            params=params,
            num_classes=len(class_names),
            random_state=config.random_state,
        )
        logger.info("Training candidate %s/%s with params=%s", candidate_index, config.search_iterations, params)
        model.fit(X_train, y_train, sample_weight=train_weights)

        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        val_metrics, _, _ = evaluate_predictions(y_true=y_val, y_pred=val_predictions, probabilities=val_probabilities, class_names=class_names)
        score = selection_score(val_metrics, config.selection_metric)
        candidate_results.append({"params": params, "validation_metrics": val_metrics, "selection_score": float(score)})
        logger.info("Candidate %s validation %s=%.4f", candidate_index, config.selection_metric, score)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
            best_val_metrics = val_metrics

    if best_model is None:
        raise RuntimeError("No candidate model was trained successfully.")

    logger.info("Best validation %s: %.4f with params=%s", config.selection_metric, best_score, best_params)

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    train_val_weights = compute_sample_weights_array(y_train_val)
    final_model = build_model(
        model_type=config.model_type,
        params=best_params,
        num_classes=len(class_names),
        random_state=config.random_state,
    )
    final_model.fit(X_train_val, y_train_val, sample_weight=train_val_weights)

    test_predictions = final_model.predict(X_test)
    test_probabilities = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None
    test_metrics, classification_report_df, confusion = evaluate_predictions(
        y_true=y_test,
        y_pred=test_predictions,
        probabilities=test_probabilities,
        class_names=class_names,
    )

    paths = training_report_paths()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, paths["model"])
    write_json(paths["class_names"], {"classes": class_names})
    write_json(paths["selected_feature_columns"], {"feature_columns": feature_columns, "feature_prefix": config.feature_prefix, "feature_set_mode": config.feature_set_mode})
    write_json(
        paths["inference_config"],
        {
            "model_type": config.model_type,
            "model_artifact": str(paths["model"]),
            "class_names_artifact": str(paths["class_names"]),
            "selected_feature_columns_artifact": str(paths["selected_feature_columns"]),
            "feature_prefix": config.feature_prefix,
            "feature_extractor_model_name": config.feature_extractor_model_name,
            "feature_vector_size": 1024,
            "missing_value_strategy": "native_model_missing_support",
            "feature_set_mode": config.feature_set_mode,
            "label_mapping": {label: index for label, index in label_to_index.items()},
        },
    )
    classification_report_df.to_csv(paths["classification_report"], index=False)
    save_confusion_matrix_image(confusion, class_names, paths["confusion_matrix"])

    split_counts = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    split_label_counts = {
        split_name: {label: int(count) for label, count in combined_df.loc[mask, "label"].value_counts().sort_index().items()}
        for split_name, mask in split_masks.items()
    }
    training_report = {
        "model_type": config.model_type,
        "feature_set_mode": config.feature_set_mode,
        "feature_column_count": int(len(feature_columns)),
        "class_names": class_names,
        "data_summary": {
            "rows_total": int(len(combined_df)),
            "dropped_rows": int(dropped_rows),
            "dataset_sources": gold_summaries,
            "split_counts": split_counts,
            "split_label_counts": split_label_counts,
            "split_info": split_info,
            "image_root": str(config.image_root) if config.image_root else None,
        },
        "imbalance_handling": {"strategy": "balanced_sample_weight", "class_weight_map": class_weight_map},
        "hyperparameter_search": {
            "selection_metric": config.selection_metric,
            "candidate_results": candidate_results,
            "best_params": best_params,
            "best_validation_score": float(best_score),
        },
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    write_json(paths["training_report"], training_report)

    summary_lines = [
        "Final Tabular Model Summary",
        f"Model type: {config.model_type}",
        f"Classes: {len(class_names)}",
        f"Feature columns: {len(feature_columns)} ({config.feature_set_mode})",
        f"Best validation {config.selection_metric}: {best_score:.4f}",
        f"Test accuracy: {test_metrics['accuracy']:.4f}",
        f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"Test macro F1: {test_metrics['macro_f1']:.4f}",
        f"Test weighted F1: {test_metrics['weighted_f1']:.4f}",
        f"Saved model: {paths['model']}",
    ]
    write_text(paths["metrics_summary"], "\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    logger.info("Training complete. Best model saved to %s", paths["model"])
    return training_report


def main() -> None:
    args = build_parser().parse_args()
    config = build_training_config(args)
    run_training(config)


if __name__ == "__main__":
    main()
