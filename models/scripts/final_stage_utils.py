from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import canonicalize_label_series, read_dataframe  # noqa: E402
from scripts.train_tabular_model import (  # noqa: E402
    TrainingConfig,
    assign_splits,
    choose_feature_columns,
    combine_gold_dataframes,
    default_gold_paths,
    normalize_dataset_name,
    standardize_gold_dataframe,
)


@dataclass
class CombinedGoldData:
    dataframe: pd.DataFrame
    numeric_features: pd.DataFrame
    feature_columns: List[str]
    class_names: List[str]
    label_to_index: Dict[str, int]
    encoded_labels: np.ndarray
    split_info: Dict[str, object]
    split_masks: Dict[str, pd.Series]
    dataset_summaries: List[Dict[str, object]]
    dropped_rows: int


def build_shared_training_config(
    *,
    image_root: Optional[Path],
    random_state: int,
    test_size: float,
    val_size: float,
    feature_set_mode: str,
    max_rows_per_dataset: Optional[int],
    use_existing_split: bool,
    feature_extractor_model_name: str = "google/vit-large-patch16-224-in21k",
) -> TrainingConfig:
    return TrainingConfig(
        image_root=image_root,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
        feature_prefix="feature_",
        feature_set_mode=feature_set_mode,
        model_type="xgboost",
        search_iterations=1,
        selection_metric="macro_f1",
        max_rows_per_dataset=max_rows_per_dataset,
        use_existing_split=use_existing_split,
        feature_extractor_model_name=feature_extractor_model_name,
    )


def load_combined_gold_data(config: TrainingConfig, logger) -> CombinedGoldData:
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

    observed_train_classes = set(np.unique(encoded_labels[split_masks["train"].to_numpy()]).tolist())
    expected_classes = set(range(len(class_names)))
    if observed_train_classes != expected_classes:
        logger.warning("Existing split omitted some classes from training. Regenerating a stratified split.")
        regenerated_config = build_shared_training_config(
            image_root=config.image_root,
            random_state=config.random_state,
            test_size=config.test_size,
            val_size=config.val_size,
            feature_set_mode=config.feature_set_mode,
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

    return CombinedGoldData(
        dataframe=combined_df,
        numeric_features=numeric_features,
        feature_columns=feature_columns,
        class_names=class_names,
        label_to_index=label_to_index,
        encoded_labels=encoded_labels,
        split_info=split_info,
        split_masks=split_masks,
        dataset_summaries=gold_summaries,
        dropped_rows=dropped_rows,
    )


def split_feature_matrices(
    combined: CombinedGoldData,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    feature_columns = list(selected_feature_columns or combined.feature_columns)
    return {
        split_name: combined.numeric_features.loc[mask, feature_columns].reset_index(drop=True)
        for split_name, mask in combined.split_masks.items()
    }


def split_labels(combined: CombinedGoldData) -> Dict[str, np.ndarray]:
    return {
        split_name: combined.encoded_labels[mask.to_numpy()]
        for split_name, mask in combined.split_masks.items()
    }
