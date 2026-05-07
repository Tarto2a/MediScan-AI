from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import ARTIFACT_REPORTS_ROOT, ensure_directories, setup_logging, write_json  # noqa: E402
from scripts.final_stage_utils import build_shared_training_config, load_combined_gold_data  # noqa: E402
from scripts.reporting_utils import (  # noqa: E402
    BRAND_COLORS,
    append_figure_descriptions,
    ensure_reporting_layout,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_distribution_grid,
    plot_ranked_horizontal_bars,
    save_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication-ready EDA and feature analysis on the merged gold-layer feature tables.")
    parser.add_argument("--image-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--feature-set-mode", choices=("union", "intersection"), default="union")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--ignore-existing-split", action="store_true")
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--plot-features", type=int, default=12)
    parser.add_argument("--correlation-plot-features", type=int, default=18)
    parser.add_argument("--correlation-report-threshold", type=float, default=0.85)
    parser.add_argument("--outlier-zscore-threshold", type=float, default=4.0)
    return parser


def detect_duplicate_feature_columns(feature_frame: pd.DataFrame) -> List[str]:
    duplicated_mask = feature_frame.T.duplicated()
    return feature_frame.columns[duplicated_mask].tolist()


def distribution_summary_frame(series: pd.Series, label_name: str) -> pd.DataFrame:
    counts = series.astype(int)
    total = float(counts.sum()) if len(counts) else 0.0
    return pd.DataFrame(
        {
            label_name: counts.index.astype(str),
            "count": counts.to_numpy(dtype=int),
            "percentage": (counts.to_numpy(dtype=float) / total) * 100.0 if total else np.zeros(len(counts)),
        }
    )


def ordered_feature_stats(statistics: pd.DataFrame, ordered_features: List[str]) -> pd.DataFrame:
    subset = statistics.loc[ordered_features, ["mean", "std", "min", "max", "skewness"]].copy()
    subset = subset.reset_index().rename(columns={"index": "feature"})
    subset["feature"] = pd.Categorical(subset["feature"], categories=ordered_features, ordered=True)
    subset = subset.sort_values("feature").reset_index(drop=True)
    subset["feature"] = subset["feature"].astype(str)
    return subset


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("run_eda.log")
    ensure_directories()
    ARTIFACT_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    layout = ensure_reporting_layout()

    config = build_shared_training_config(
        image_root=Path(args.image_root).resolve() if args.image_root else None,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        feature_set_mode=args.feature_set_mode,
        max_rows_per_dataset=args.max_rows_per_dataset,
        use_existing_split=not args.ignore_existing_split,
    )
    combined = load_combined_gold_data(config=config, logger=logger)

    dataframe = combined.dataframe.copy()
    feature_frame = combined.numeric_features.copy()
    split_masks = combined.split_masks
    train_features = feature_frame.loc[split_masks["train"]].reset_index(drop=True)
    train_labels = combined.encoded_labels[split_masks["train"].to_numpy()]

    label_distribution = dataframe["label"].value_counts().sort_values(ascending=False)
    dataset_distribution = dataframe["dataset_name"].fillna("Unknown").value_counts().sort_values(ascending=False)
    split_distribution = dataframe["split"].fillna("unknown").value_counts().sort_index()
    feature_types = {
        "numeric_columns": int(len(feature_frame.columns)),
        "metadata_columns": int(dataframe.shape[1] - feature_frame.shape[1]),
        "dtype_counts": {str(key): int(value) for key, value in dataframe.dtypes.astype(str).value_counts().items()},
    }

    statistics = feature_frame.describe().transpose()
    statistics["variance"] = feature_frame.var(numeric_only=True)
    statistics["skewness"] = feature_frame.skew(numeric_only=True)
    statistics["missing_rate"] = feature_frame.isna().mean()
    statistics = statistics.sort_index()

    statistics_path = layout["tables_root"] / "statistics_summary.csv"
    save_table(statistics.reset_index().rename(columns={"index": "feature"}), statistics_path)
    statistics.reset_index().rename(columns={"index": "feature"}).to_csv(ARTIFACT_REPORTS_ROOT / "statistics_summary.csv", index=False)

    label_distribution_summary = distribution_summary_frame(label_distribution, "label")
    dataset_distribution_summary = distribution_summary_frame(dataset_distribution, "dataset_name")
    split_distribution_summary = distribution_summary_frame(split_distribution, "split")
    save_table(label_distribution_summary, layout["tables_root"] / "label_distribution_summary.csv")
    save_table(dataset_distribution_summary, layout["tables_root"] / "dataset_distribution_summary.csv")
    save_table(split_distribution_summary, layout["tables_root"] / "split_distribution_summary.csv")

    missing_rates = feature_frame.isna().mean().sort_values(ascending=False)
    duplicated_rows = int(dataframe.duplicated().sum())
    duplicated_feature_columns = detect_duplicate_feature_columns(feature_frame)
    variance_series = statistics["variance"].fillna(0.0)
    constant_features = variance_series[variance_series <= 0.0].index.tolist()
    near_constant_features = variance_series[variance_series <= 1e-6].index.tolist()
    invalid_numeric_values = int(np.isinf(feature_frame.to_numpy(dtype=np.float32, copy=True)).sum())

    high_variance_features = variance_series.sort_values(ascending=False).head(max(args.correlation_plot_features, 100))
    top_outlier_features = high_variance_features.head(100).index.tolist()
    outlier_frame = feature_frame[top_outlier_features]
    outlier_means = outlier_frame.mean()
    outlier_stds = outlier_frame.std(ddof=0).replace(0.0, np.nan)
    outlier_zscores = (outlier_frame - outlier_means) / outlier_stds
    outlier_rates = outlier_zscores.abs().gt(args.outlier_zscore_threshold).mean().fillna(0.0).sort_values(ascending=False)

    imputed_train = train_features.fillna(train_features.median()).fillna(0.0)
    mutual_info_scores = mutual_info_classif(imputed_train, train_labels, random_state=args.random_state)
    mutual_info_series = pd.Series(mutual_info_scores, index=train_features.columns).sort_values(ascending=False)
    top_score_features = mutual_info_series.head(args.top_features)
    representative_features = top_score_features.head(args.plot_features).index.tolist()
    representative_feature_stats = ordered_feature_stats(statistics, representative_features)
    representative_feature_stats["std"] = representative_feature_stats["std"].fillna(0.0)
    save_table(representative_feature_stats, layout["tables_root"] / "representative_feature_statistics.csv")

    variance_values = variance_series.sort_values(ascending=False).reset_index()
    variance_values.columns = ["feature", "variance"]
    mutual_information_values = mutual_info_series.reset_index()
    mutual_information_values.columns = ["feature", "mutual_information"]
    save_table(variance_values, layout["tables_root"] / "variance_values.csv")
    save_table(mutual_information_values, layout["tables_root"] / "mutual_information_values.csv")

    top_corr_features = mutual_info_series.head(args.correlation_plot_features).index.tolist()
    corr_frame = feature_frame[top_corr_features].fillna(feature_frame[top_corr_features].median()).fillna(0.0)
    corr_matrix = corr_frame.corr().fillna(0.0)
    corr_matrix_abs = corr_matrix.abs()
    upper_triangle = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
    high_correlation_pairs: List[Dict[str, object]] = []
    for column in upper_triangle.columns:
        matches = upper_triangle[column][upper_triangle[column] >= args.correlation_report_threshold]
        for partner, value in matches.items():
            high_correlation_pairs.append(
                {
                    "feature_a": str(partner),
                    "feature_b": str(column),
                    "absolute_correlation": float(value),
                    "signed_correlation": float(corr_matrix.loc[partner, column]),
                }
            )
    high_correlation_pairs_df = pd.DataFrame(high_correlation_pairs)
    if high_correlation_pairs_df.empty:
        high_correlation_pairs_df = pd.DataFrame(columns=["feature_a", "feature_b", "absolute_correlation", "signed_correlation"])
    else:
        high_correlation_pairs_df = high_correlation_pairs_df.sort_values("absolute_correlation", ascending=False).reset_index(drop=True)
    save_table(high_correlation_pairs_df, layout["tables_root"] / "high_correlation_pairs.csv")

    eda_summary = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_columns": int(feature_frame.shape[1]),
        "metadata_columns": int(dataframe.shape[1] - feature_frame.shape[1]),
        "class_count": int(len(combined.class_names)),
        "class_names": combined.class_names,
        "label_distribution": {str(key): int(value) for key, value in label_distribution.items()},
        "dataset_distribution": {str(key): int(value) for key, value in dataset_distribution.items()},
        "split_distribution": {str(key): int(value) for key, value in split_distribution.items()},
        "feature_types": feature_types,
        "dataset_sources": combined.dataset_summaries,
        "split_info": combined.split_info,
        "dropped_rows": int(combined.dropped_rows),
        "artifacts_root": str(layout["plots_root"]),
        "tables_root": str(layout["tables_root"]),
    }
    write_json(ARTIFACT_REPORTS_ROOT / "eda_summary.json", eda_summary)

    data_quality_report = {
        "missing_values": {
            "total_missing_cells": int(feature_frame.isna().sum().sum()),
            "features_with_missing_values": int((missing_rates > 0).sum()),
            "top_missing_rates": {str(key): float(value) for key, value in missing_rates.head(args.top_features).items()},
        },
        "duplicates": {
            "duplicated_rows": duplicated_rows,
            "duplicated_feature_columns": duplicated_feature_columns,
            "duplicated_feature_column_count": int(len(duplicated_feature_columns)),
        },
        "variance_checks": {
            "constant_feature_count": int(len(constant_features)),
            "near_constant_feature_count": int(len(near_constant_features)),
            "constant_feature_examples": constant_features[:20],
            "near_constant_feature_examples": near_constant_features[:20],
        },
        "outliers": {
            "zscore_threshold": args.outlier_zscore_threshold,
            "top_outlier_rates": {str(key): float(value) for key, value in outlier_rates.head(args.top_features).items()},
        },
        "invalid_numeric_values": invalid_numeric_values,
        "labels": {
            "class_names": combined.class_names,
            "label_count": int(len(combined.class_names)),
        },
    }
    write_json(ARTIFACT_REPORTS_ROOT / "data_quality_report.json", data_quality_report)

    feature_analysis_report = {
        "top_mutual_information_features": {str(key): float(value) for key, value in top_score_features.items()},
        "top_variance_features": {str(key): float(value) for key, value in variance_series.sort_values(ascending=False).head(args.top_features).items()},
        "high_correlation_pairs": high_correlation_pairs_df.head(50).to_dict(orient="records"),
        "feature_space": {
            "initial_feature_count": int(len(combined.feature_columns)),
            "top_features_considered_for_correlation_plot": int(len(top_corr_features)),
        },
        "analysis_method": {
            "target_association_metric": "mutual_info_classif_on_train_split",
            "correlation_metric": "pearson_correlation",
            "high_correlation_reporting_threshold": float(args.correlation_report_threshold),
        },
        "tables": {
            "statistics_summary": str(statistics_path),
            "variance_values": str(layout["tables_root"] / "variance_values.csv"),
            "mutual_information_values": str(layout["tables_root"] / "mutual_information_values.csv"),
            "high_correlation_pairs": str(layout["tables_root"] / "high_correlation_pairs.csv"),
            "representative_feature_statistics": str(layout["tables_root"] / "representative_feature_statistics.csv"),
        },
    }
    write_json(ARTIFACT_REPORTS_ROOT / "feature_analysis_report.json", feature_analysis_report)

    figure_descriptions: List[tuple[Path, str]] = []

    label_distribution_plot = layout["eda"] / "class_distribution_multiclass.png"
    plot_categorical_distribution(
        label_distribution_summary,
        label_column="label",
        count_column="count",
        output_paths=[label_distribution_plot, ARTIFACT_REPORTS_ROOT / "label_distribution.png"],
        title="Class Distribution of the Multi-Class Dataset",
        subtitle="Counts are shown above each bar to highlight the imbalance across disease categories.",
        ylabel="Image Count",
        color=BRAND_COLORS["teal"],
        rotate_xticks=35,
    )
    figure_descriptions.append((label_distribution_plot, "Class imbalance across the full multi-class dataset with exact counts above each category bar."))

    dataset_distribution_plot = layout["eda"] / "dataset_source_distribution.png"
    plot_categorical_distribution(
        dataset_distribution_summary,
        label_column="dataset_name",
        count_column="count",
        output_paths=[dataset_distribution_plot, ARTIFACT_REPORTS_ROOT / "dataset_source_distribution.png"],
        title="Dataset Contribution Across the Combined Gold-Layer Cohort",
        subtitle="The merged modeling dataset is built from NIH, TCIA, and RSNA contributions.",
        ylabel="Row Count",
        color=BRAND_COLORS["plum"],
        rotate_xticks=0,
    )
    figure_descriptions.append((dataset_distribution_plot, "Relative sample contribution of each source dataset in the merged modeling cohort."))

    split_distribution_plot = layout["eda"] / "split_distribution.png"
    plot_categorical_distribution(
        split_distribution_summary,
        label_column="split",
        count_column="count",
        output_paths=split_distribution_plot,
        title="Train, Validation, and Test Split Sizes",
        subtitle="Split counts for the final merged Gold-layer dataset used in modeling and evaluation.",
        ylabel="Row Count",
        color=BRAND_COLORS["amber"],
        rotate_xticks=0,
    )
    figure_descriptions.append((split_distribution_plot, "Final train, validation, and test sample counts used in the downstream FT-Transformer workflow."))

    missing_values_plot = layout["eda"] / "missing_value_rates_top_features.png"
    plot_ranked_horizontal_bars(
        missing_rates.sort_values(ascending=False).head(args.top_features),
        output_paths=[missing_values_plot, ARTIFACT_REPORTS_ROOT / "missing_values_summary.png"],
        title="Top Features by Missing-Value Rate",
        subtitle="The highest-missing features help reveal sparsity patterns before final-stage modeling.",
        xlabel="Missing Rate",
        color=BRAND_COLORS["amber"],
        value_fmt="{:.1%}",
    )
    figure_descriptions.append((missing_values_plot, "Top feature columns ranked by missing-value rate before final-stage feature selection."))

    correlation_heatmap_plot = layout["feature_analysis"] / "feature_correlation_heatmap_top_selected_features.png"
    plot_correlation_heatmap(
        corr_matrix,
        output_paths=[correlation_heatmap_plot, ARTIFACT_REPORTS_ROOT / "correlation_heatmap.png"],
        title="Feature Correlation Heatmap (Top Selected Features)",
        subtitle="This figure highlights relationships between the most important selected features and helps detect redundancy.",
        high_correlation_pairs=high_correlation_pairs_df,
    )
    figure_descriptions.append((correlation_heatmap_plot, "Pearson correlation structure for the most informative features, used to inspect redundancy and feature overlap."))

    variance_plot = layout["feature_analysis"] / "top_features_by_variance.png"
    plot_ranked_horizontal_bars(
        variance_series.sort_values(ascending=False).head(args.top_features),
        output_paths=[variance_plot, ARTIFACT_REPORTS_ROOT / "feature_variance_top20.png"],
        title="Top Features by Variance (Data Spread Importance)",
        subtitle="Higher variance can indicate stronger discriminatory potential by capturing broader feature spread across samples.",
        xlabel="Variance",
        color=BRAND_COLORS["teal"],
        value_fmt="{:.4f}",
    )
    figure_descriptions.append((variance_plot, "Highest-variance feature dimensions, showing which ViT-derived features vary most across the cohort."))

    mutual_information_plot = layout["feature_analysis"] / "top_features_by_mutual_information.png"
    plot_ranked_horizontal_bars(
        top_score_features,
        output_paths=[mutual_information_plot, ARTIFACT_REPORTS_ROOT / "top_feature_scores.png"],
        title="Top Features by Mutual Information (Relevance to Target Class)",
        subtitle="Mutual information scores quantify how strongly each feature is associated with the diagnosis label.",
        xlabel="Mutual Information Score",
        color=BRAND_COLORS["coral"],
        value_fmt="{:.4f}",
    )
    figure_descriptions.append((mutual_information_plot, "Most label-informative features ranked by mutual information on the training split."))

    representative_distribution_plot = layout["feature_analysis"] / "representative_feature_distributions.png"
    plot_distribution_grid(
        feature_frame[representative_features],
        statistics_frame=statistics,
        output_paths=[representative_distribution_plot, ARTIFACT_REPORTS_ROOT / "feature_distributions.png"],
        title="Representative Distributions of ViT-Extracted Features",
        subtitle="Each subplot includes mean, standard deviation, minimum, and maximum values to support paper-ready interpretation.",
    )
    figure_descriptions.append((representative_distribution_plot, "Representative feature histograms with embedded descriptive statistics for publication-ready interpretation."))

    append_figure_descriptions(figure_descriptions)

    summary_payload = {
        "eda_summary": str(ARTIFACT_REPORTS_ROOT / "eda_summary.json"),
        "statistics_summary": str(statistics_path),
        "data_quality_report": str(ARTIFACT_REPORTS_ROOT / "data_quality_report.json"),
        "feature_analysis_report": str(ARTIFACT_REPORTS_ROOT / "feature_analysis_report.json"),
        "plots_root": str(layout["plots_root"]),
        "tables_root": str(layout["tables_root"]),
    }
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
