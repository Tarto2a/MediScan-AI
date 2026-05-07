from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from scripts.common import (
    ARTIFACT_PLOTS_ROOT,
    ARTIFACT_TABLES_ROOT,
    PROJECT_ROOT,
    ensure_directories,
    write_text,
)

PLOT_DPI = 300

BRAND_COLORS: Dict[str, str] = {
    "navy": "#12355B",
    "blue": "#2F6690",
    "sky": "#6FA3C8",
    "teal": "#1B998B",
    "green": "#3A7D44",
    "amber": "#E09F3E",
    "coral": "#D1495B",
    "plum": "#7A306C",
    "slate": "#4F5D75",
    "light": "#F5F7FA",
    "grid": "#D9E2EC",
    "text": "#1F2933",
}

MODEL_COLORS: Dict[str, str] = {
    "FT-Transformer": BRAND_COLORS["navy"],
    "XGBoost": BRAND_COLORS["coral"],
    "RandomForest": BRAND_COLORS["green"],
    "LogisticRegression": BRAND_COLORS["amber"],
    "MLP": BRAND_COLORS["plum"],
}


def ensure_reporting_layout() -> Dict[str, Path]:
    ensure_directories()
    directories = {
        "plots_root": ARTIFACT_PLOTS_ROOT,
        "eda": ARTIFACT_PLOTS_ROOT / "eda",
        "feature_analysis": ARTIFACT_PLOTS_ROOT / "feature_analysis",
        "training": ARTIFACT_PLOTS_ROOT / "training",
        "benchmarks": ARTIFACT_PLOTS_ROOT / "benchmarks",
        "confusion_matrices": ARTIFACT_PLOTS_ROOT / "confusion_matrices",
        "binary": ARTIFACT_PLOTS_ROOT / "binary",
        "evaluation": ARTIFACT_PLOTS_ROOT / "evaluation",
        "tables_root": ARTIFACT_TABLES_ROOT,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": PLOT_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": BRAND_COLORS["text"],
            "axes.edgecolor": BRAND_COLORS["grid"],
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.color": BRAND_COLORS["text"],
            "ytick.color": BRAND_COLORS["text"],
            "grid.color": BRAND_COLORS["grid"],
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.autolayout": False,
        }
    )


def _normalize_output_paths(output_paths: Path | Sequence[Path]) -> List[Path]:
    if isinstance(output_paths, Path):
        return [output_paths]
    return list(output_paths)


def save_figure(fig: plt.Figure, output_paths: Path | Sequence[Path]) -> None:
    for output_path in _normalize_output_paths(output_paths):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def add_title_and_subtitle(
    fig: plt.Figure,
    *,
    title: str,
    subtitle: Optional[str] = None,
    y_title: float = 0.98,
    y_subtitle: float = 0.95,
) -> None:
    fig.suptitle(title, y=y_title, fontsize=18, fontweight="bold", color=BRAND_COLORS["text"])
    if subtitle:
        fig.text(
            0.5,
            y_subtitle,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            color=BRAND_COLORS["slate"],
        )


def annotate_bar_values(
    ax: plt.Axes,
    *,
    orientation: str = "vertical",
    fmt: str = "{:.0f}",
    padding: float = 0.02,
) -> None:
    if orientation == "horizontal":
        max_value = max((patch.get_width() for patch in ax.patches), default=1.0)
        for patch in ax.patches:
            value = patch.get_width()
            ax.text(
                value + max_value * padding,
                patch.get_y() + patch.get_height() / 2.0,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=10,
                color=BRAND_COLORS["text"],
            )
        return

    max_value = max((patch.get_height() for patch in ax.patches), default=1.0)
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            value + max_value * padding,
            fmt.format(value),
            va="bottom",
            ha="center",
            fontsize=10,
            color=BRAND_COLORS["text"],
        )


def plot_categorical_distribution(
    summary_frame: pd.DataFrame,
    *,
    label_column: str,
    count_column: str,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    ylabel: str = "Count",
    color: str = BRAND_COLORS["blue"],
    rotate_xticks: int = 30,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(
        summary_frame[label_column].astype(str),
        summary_frame[count_column].astype(float),
        color=color,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=rotate_xticks)
    annotate_bar_values(ax, fmt="{:.0f}", padding=0.01)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.90))
    save_figure(fig, output_paths)


def plot_ranked_horizontal_bars(
    series: pd.Series,
    *,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    xlabel: str,
    color: str,
    value_fmt: str = "{:.4f}",
) -> None:
    apply_publication_style()
    ordered = series.sort_values(ascending=True)
    fig_height = max(6, 0.45 * len(ordered) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(ordered.index.astype(str), ordered.to_numpy(dtype=float), color=color, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    annotate_bar_values(ax, orientation="horizontal", fmt=value_fmt, padding=0.015)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    *,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    high_correlation_pairs: Optional[pd.DataFrame] = None,
    annotate_limit: int = 18,
) -> None:
    apply_publication_style()
    cmap = LinearSegmentedColormap.from_list(
        "publication_diverging",
        [BRAND_COLORS["navy"], "#F7FBFF", BRAND_COLORS["coral"]],
        N=256,
    )
    size = max(12, min(18, 0.45 * len(correlation.columns)))
    fig, ax = plt.subplots(figsize=(size, size))
    matrix = correlation.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_yticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(correlation.columns, fontsize=9)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")

    if len(correlation.columns) <= annotate_limit:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > 0.55 else BRAND_COLORS["text"],
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson Correlation", rotation=270, labelpad=18)

    note = None
    if high_correlation_pairs is not None and not high_correlation_pairs.empty:
        note = f"Highly correlated pairs above threshold: {len(high_correlation_pairs)}"

    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=10, color=BRAND_COLORS["slate"])
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_distribution_grid(
    dataframe: pd.DataFrame,
    *,
    statistics_frame: pd.DataFrame,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    feature_label_suffix: str = "ViT-Derived Feature",
) -> None:
    apply_publication_style()
    columns = list(dataframe.columns)
    if not columns:
        return

    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(5.5 * nrows, 6)))
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, column in zip(axes, columns):
        values = dataframe[column].dropna().to_numpy(dtype=float)
        axis.hist(values, bins=28, color=BRAND_COLORS["blue"], edgecolor="white", alpha=0.92)
        axis.set_title(f"Distribution of {column} ({feature_label_suffix})", fontsize=12.5, pad=10)
        axis.set_xlabel("Feature Value")
        axis.set_ylabel("Sample Count")
        stats = statistics_frame.loc[column]
        summary_text = "\n".join(
            [
                f"mean: {float(stats['mean']):.3f}",
                f"std: {float(stats['std']):.3f}",
                f"min: {float(stats['min']):.3f}",
                f"max: {float(stats['max']):.3f}",
            ]
        )
        axis.text(
            0.98,
            0.97,
            summary_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": BRAND_COLORS["grid"]},
        )

    for axis in axes[len(columns):]:
        axis.axis("off")

    add_title_and_subtitle(fig, title=title, subtitle=subtitle, y_title=0.995, y_subtitle=0.965)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    save_figure(fig, output_paths)


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    class_names: Sequence[str],
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    normalize: bool,
    cmap: str = "Blues",
) -> None:
    apply_publication_style()
    display_matrix = matrix.astype(float)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(display_matrix, row_sums, where=row_sums != 0)

    fig_width = max(10, 0.75 * len(class_names) + 6)
    fig_height = max(8, 0.65 * len(class_names) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(display_matrix, cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    threshold = float(display_matrix.max()) * 0.55 if display_matrix.size else 0.0
    for row_idx in range(display_matrix.shape[0]):
        for col_idx in range(display_matrix.shape[1]):
            value = display_matrix[row_idx, col_idx]
            text = f"{value:.2f}" if normalize else f"{int(matrix[row_idx, col_idx])}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else BRAND_COLORS["text"],
                fontsize=10,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized Count" if normalize else "Count", rotation=270, labelpad=18)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_training_history(
    history_frame: pd.DataFrame,
    *,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    best_epoch: int,
    early_stop_epoch: Optional[int],
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    loss_axis, metric_axis = axes

    loss_axis.plot(history_frame["epoch"], history_frame["train_loss"], label="Train Loss", color=BRAND_COLORS["blue"], linewidth=2.2)
    loss_axis.plot(history_frame["epoch"], history_frame["val_loss"], label="Validation Loss", color=BRAND_COLORS["coral"], linewidth=2.2)
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend(loc="upper right")

    metric_axis.plot(
        history_frame["epoch"],
        history_frame["val_macro_f1"],
        label="Validation Macro F1",
        color=BRAND_COLORS["green"],
        linewidth=2.4,
    )
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("Macro F1")
    metric_axis.legend(loc="lower right")

    best_row = history_frame.loc[history_frame["epoch"] == best_epoch].head(1)
    if not best_row.empty:
        best_val_f1 = float(best_row["val_macro_f1"].iloc[0])
        for axis in axes:
            axis.axvline(best_epoch, color=BRAND_COLORS["plum"], linestyle="--", linewidth=1.4, alpha=0.95)
        metric_axis.scatter([best_epoch], [best_val_f1], color=BRAND_COLORS["plum"], s=60, zorder=5)
        metric_axis.annotate(
            f"Best epoch: {best_epoch}\nVal Macro F1: {best_val_f1:.4f}",
            xy=(best_epoch, best_val_f1),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": BRAND_COLORS["grid"]},
        )

    if early_stop_epoch is not None:
        for axis in axes:
            axis.axvline(early_stop_epoch, color=BRAND_COLORS["amber"], linestyle=":", linewidth=1.5, alpha=0.95)
        metric_axis.annotate(
            f"Early stop: {early_stop_epoch}",
            xy=(early_stop_epoch, float(history_frame["val_macro_f1"].iloc[-1])),
            xytext=(12, -28),
            textcoords="offset points",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF8E8", "edgecolor": BRAND_COLORS["amber"]},
            color=BRAND_COLORS["slate"],
        )

    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_grouped_metric_bars(
    metrics_frame: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    apply_publication_style()
    pretty_names = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "roc_auc": "ROC AUC",
    }
    metric_labels = [pretty_names.get(column, column) for column in metric_columns]
    x_positions = np.arange(len(metric_columns), dtype=float)
    width = 0.14 if len(metrics_frame) >= 4 else 0.18

    fig, ax = plt.subplots(figsize=(15, 7))
    for index, (_, row) in enumerate(metrics_frame.iterrows()):
        offsets = x_positions + (index - (len(metrics_frame) - 1) / 2.0) * width
        values = [float(row[column]) for column in metric_columns]
        color = MODEL_COLORS.get(str(row["model"]), BRAND_COLORS["slate"])
        ax.bar(offsets, values, width=width, label=str(row["model"]), color=color, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(metrics_frame[list(metric_columns)].max().max()) + 0.08)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(3, len(metrics_frame)))
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_metric_highlight(
    metrics_frame: pd.DataFrame,
    *,
    metric_column: str,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    apply_publication_style()
    ordered = metrics_frame.sort_values(metric_column, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [MODEL_COLORS.get(model, BRAND_COLORS["slate"]) for model in ordered["model"].astype(str)]
    ax.barh(ordered["model"].astype(str), ordered[metric_column].astype(float), color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(metric_column.replace("_", " ").title())
    ax.set_ylabel("Model")
    annotate_bar_values(ax, orientation="horizontal", fmt="{:.4f}", padding=0.015)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_metric_tradeoff(
    metrics_frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for _, row in metrics_frame.iterrows():
        model_name = str(row["model"])
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        color = MODEL_COLORS.get(model_name, BRAND_COLORS["slate"])
        ax.scatter(x_value, y_value, s=110 if model_name == "FT-Transformer" else 85, color=color, edgecolor="white", linewidth=0.8)
        ax.text(x_value + 0.002, y_value + 0.002, model_name, fontsize=10, color=BRAND_COLORS["text"])

    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(y_column.replace("_", " ").title())
    ax.grid(True)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_feature_selection_counts(
    *,
    before_count: int,
    after_count: int,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    apply_publication_style()
    comparison_frame = pd.DataFrame(
        {
            "stage": ["Before Final Selection", "After Final Selection"],
            "count": [before_count, after_count],
        }
    )
    plot_categorical_distribution(
        comparison_frame,
        label_column="stage",
        count_column="count",
        output_paths=output_paths,
        title=title,
        subtitle=subtitle,
        ylabel="Feature Count",
        color=BRAND_COLORS["teal"],
        rotate_xticks=0,
    )


def plot_binary_roc_comparison(
    curve_records: Sequence[Dict[str, object]],
    *,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.5, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color=BRAND_COLORS["slate"], linewidth=1.2, label="Chance")
    for record in curve_records:
        model_name = str(record["model"])
        ax.plot(
            record["fpr"],
            record["tpr"],
            linewidth=2.2,
            color=MODEL_COLORS.get(model_name, BRAND_COLORS["slate"]),
            label=f"{model_name} (AUC={float(record['roc_auc']):.4f})",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)


def plot_multiclass_ovr_curves(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
    curve_type: str,
) -> pd.DataFrame:
    apply_publication_style()
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    if y_true_bin.shape[1] != len(class_names) or np.any(y_true_bin.sum(axis=0) == 0):
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(11.5, 8))
    summary_rows: List[Dict[str, object]] = []
    palette = list(MODEL_COLORS.values()) + [BRAND_COLORS["sky"], BRAND_COLORS["amber"], BRAND_COLORS["teal"], BRAND_COLORS["slate"]]

    for class_index, class_name in enumerate(class_names):
        color = palette[class_index % len(palette)]
        if curve_type == "roc":
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], probabilities[:, class_index])
            score_value = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, linewidth=2.0, color=color, label=f"{class_name} (AUC={score_value:.4f})")
            summary_rows.append({"model": model_name, "class_name": class_name, "roc_auc": float(score_value)})
        else:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, class_index], probabilities[:, class_index])
            score_value = average_precision_score(y_true_bin[:, class_index], probabilities[:, class_index])
            ax.plot(recall, precision, linewidth=2.0, color=color, label=f"{class_name} (AP={score_value:.4f})")
            summary_rows.append({"model": model_name, "class_name": class_name, "average_precision": float(score_value)})

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color=BRAND_COLORS["slate"], linewidth=1.2, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    ax.legend(loc="lower left" if curve_type == "pr" else "lower right", fontsize=9)
    add_title_and_subtitle(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    save_figure(fig, output_paths)
    return pd.DataFrame(summary_rows)


def plot_per_class_recall(
    report_frame: pd.DataFrame,
    *,
    model_name: str,
    output_paths: Path | Sequence[Path],
    title: str,
    subtitle: Optional[str],
) -> None:
    per_class_frame = report_frame.loc[~report_frame["label"].isin(["accuracy", "macro avg", "weighted avg"])].copy()
    if per_class_frame.empty:
        return
    series = per_class_frame.set_index("label")["recall"].astype(float).sort_values(ascending=False)
    plot_ranked_horizontal_bars(
        series,
        output_paths=output_paths,
        title=title,
        subtitle=subtitle or f"Per-class recall scores for {model_name}.",
        xlabel="Recall",
        color=MODEL_COLORS.get(model_name, BRAND_COLORS["navy"]),
        value_fmt="{:.3f}",
    )


def summarize_top_misclassifications(
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for index, class_name in enumerate(class_names):
        total = int(matrix[index].sum())
        correct = int(matrix[index, index])
        off_diagonal = matrix[index].copy()
        off_diagonal[index] = 0
        top_misclassified_index = int(np.argmax(off_diagonal)) if off_diagonal.size else index
        top_misclassified_count = int(off_diagonal[top_misclassified_index]) if off_diagonal.size else 0
        rows.append(
            {
                "true_label": class_name,
                "support": total,
                "correct_predictions": correct,
                "misclassified_count": total - correct,
                "misclassification_rate": float((total - correct) / total) if total else 0.0,
                "most_common_incorrect_prediction": class_names[top_misclassified_index] if top_misclassified_count else "",
                "most_common_incorrect_count": top_misclassified_count,
            }
        )
    return pd.DataFrame(rows).sort_values(["misclassification_rate", "misclassified_count"], ascending=[False, False]).reset_index(drop=True)


def append_figure_descriptions(
    entries: Iterable[Tuple[Path, str]],
    output_path: Optional[Path] = None,
) -> None:
    destination = output_path or (ARTIFACT_PLOTS_ROOT / "figure_descriptions.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: List[str] = []
    if destination.exists():
        existing_lines = [line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()]

    seen = set(existing_lines)
    new_lines = list(existing_lines)
    for path, description in entries:
        try:
            relative_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_path = path
        line = f"{relative_path} - {description}"
        if line not in seen:
            seen.add(line)
            new_lines.append(line)

    write_text(destination, "\n".join(new_lines) + ("\n" if new_lines else ""))
