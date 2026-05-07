from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    ARTIFACT_REPORTS_ROOT,
    MODELS_ROOT,
    ensure_directories,
    setup_logging,
    write_json,
    write_text,
)
from scripts.final_stage_utils import (  # noqa: E402
    build_shared_training_config,
    load_combined_gold_data,
    split_feature_matrices,
    split_labels,
)
from scripts.reporting_utils import (  # noqa: E402
    BRAND_COLORS,
    MODEL_COLORS,
    append_figure_descriptions,
    ensure_reporting_layout,
    plot_binary_roc_comparison,
    plot_categorical_distribution,
    plot_confusion_matrix,
    plot_feature_selection_counts,
    plot_grouped_metric_bars,
    plot_metric_highlight,
    plot_metric_tradeoff,
    plot_multiclass_ovr_curves,
    plot_per_class_recall,
    plot_ranked_horizontal_bars,
    plot_training_history,
    save_table,
    summarize_top_misclassifications,
)
from scripts.train_tabular_model import evaluate_predictions  # noqa: E402


@dataclass
class FTTrainingArgs:
    image_root: Optional[Path]
    feature_set_mode: str
    random_state: int
    test_size: float
    val_size: float
    max_rows_per_dataset: Optional[int]
    use_existing_split: bool
    missing_threshold: float
    near_constant_threshold: float
    correlation_threshold: float
    mutual_info_threshold: float
    max_selected_features: Optional[int]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    patience: int
    scheduler: str
    selection_metric: str
    class_weight_mode: str


@dataclass
class ModelEvaluationBundle:
    model_name: str
    metrics: Dict[str, object]
    report_df: pd.DataFrame
    confusion: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: Optional[np.ndarray]


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        num_classes: int,
        d_token: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features=n_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(inputs)
        cls = self.cls_token.expand(inputs.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        cls_representation = self.norm(encoded[:, 0])
        return self.head(cls_representation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FT-Transformer and create publication-ready benchmark and evaluation reports.")
    parser.add_argument("--image-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--feature-set-mode", choices=("union", "intersection"), default="union")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--ignore-existing-split", action="store_true")
    parser.add_argument("--missing-threshold", type=float, default=0.35)
    parser.add_argument("--near-constant-threshold", type=float, default=1e-6)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--mutual-info-threshold", type=float, default=0.001)
    parser.add_argument("--max-selected-features", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler", choices=("plateau", "none"), default="plateau")
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"), default="macro_f1")
    parser.add_argument("--class-weight-mode", choices=("balanced", "none"), default="balanced")
    return parser


def parse_args() -> FTTrainingArgs:
    args = build_parser().parse_args()
    return FTTrainingArgs(
        image_root=Path(args.image_root).resolve() if args.image_root else None,
        feature_set_mode=args.feature_set_mode,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        max_rows_per_dataset=args.max_rows_per_dataset,
        use_existing_split=not args.ignore_existing_split,
        missing_threshold=args.missing_threshold,
        near_constant_threshold=args.near_constant_threshold,
        correlation_threshold=args.correlation_threshold,
        mutual_info_threshold=args.mutual_info_threshold,
        max_selected_features=args.max_selected_features,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
        scheduler=args.scheduler,
        selection_metric=args.selection_metric,
        class_weight_mode=args.class_weight_mode,
    )


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_paths(layout: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
    report_layout = layout or ensure_reporting_layout()
    return {
        "model": MODELS_ROOT / "best_ft_transformer.pth",
        "label_mapping": MODELS_ROOT / "label_mapping.json",
        "selected_features": MODELS_ROOT / "selected_features.json",
        "preprocessing_config": MODELS_ROOT / "preprocessing_config.json",
        "preprocessor": MODELS_ROOT / "ft_preprocessor.joblib",
        "feature_selection_report": ARTIFACT_REPORTS_ROOT / "feature_selection_report.json",
        "feature_selection_details": ARTIFACT_REPORTS_ROOT / "feature_selection_details.csv",
        "benchmark_results": ARTIFACT_REPORTS_ROOT / "benchmark_results.csv",
        "final_metrics": ARTIFACT_REPORTS_ROOT / "final_metrics.json",
        "classification_report": ARTIFACT_REPORTS_ROOT / "ft_classification_report.csv",
        "confusion_matrix": ARTIFACT_REPORTS_ROOT / "confusion_matrix.png",
        "training_history": ARTIFACT_REPORTS_ROOT / "ft_training_history.png",
        "feature_importance_plot": ARTIFACT_REPORTS_ROOT / "top_feature_importance_plot.png",
        "metrics_summary": ARTIFACT_REPORTS_ROOT / "metrics_summary.txt",
        "plots_root": report_layout["plots_root"],
        "tables_root": report_layout["tables_root"],
        "training_history_csv": report_layout["tables_root"] / "training_history.csv",
        "selected_feature_statistics": report_layout["tables_root"] / "selected_feature_statistics.csv",
        "per_model_metrics": report_layout["tables_root"] / "per_model_metrics_multiclass.csv",
        "per_class_metrics": report_layout["tables_root"] / "per_class_metrics_multiclass.csv",
        "binary_label_distribution_summary": report_layout["tables_root"] / "binary_label_distribution_summary.csv",
        "binary_metrics_summary": report_layout["tables_root"] / "binary_metrics_summary.csv",
        "binary_per_class_metrics": report_layout["tables_root"] / "binary_per_class_metrics.csv",
        "feature_importance_values": report_layout["tables_root"] / "feature_importance_values.csv",
        "top_misclassified_classes": report_layout["tables_root"] / "top_misclassified_classes.csv",
        "multiclass_ovr_roc_summary": report_layout["tables_root"] / "ft_transformer_multiclass_ovr_roc_summary.csv",
        "multiclass_ovr_pr_summary": report_layout["tables_root"] / "ft_transformer_multiclass_precision_recall_summary.csv",
        "figure_descriptions": report_layout["plots_root"] / "figure_descriptions.txt",
    }


def fit_feature_selector(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    args: FTTrainingArgs,
) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    missing_rates = X_train.isna().mean()
    removed_records: List[Dict[str, object]] = []
    kept_after_missing = [column for column in X_train.columns if float(missing_rates[column]) <= args.missing_threshold]
    for column in X_train.columns:
        if column not in kept_after_missing:
            removed_records.append({"feature": column, "reason": "high_missing", "score": float(missing_rates[column])})

    filtered = X_train[kept_after_missing].copy()
    median_values = filtered.median()
    imputed = filtered.fillna(median_values)

    variances = imputed.var().fillna(0.0)
    kept_after_variance = [column for column in imputed.columns if float(variances[column]) > args.near_constant_threshold]
    for column in imputed.columns:
        if column not in kept_after_variance:
            removed_records.append({"feature": column, "reason": "near_constant", "score": float(variances[column])})

    variance_filtered = imputed[kept_after_variance].copy()
    corr_matrix = variance_filtered.corr().abs().fillna(0.0)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_to_drop: List[str] = []
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] >= args.correlation_threshold]
        if not correlated.empty:
            correlated_to_drop.append(column)
            strongest_partner = correlated.sort_values(ascending=False).index[0]
            removed_records.append(
                {
                    "feature": column,
                    "reason": "high_correlation",
                    "score": float(correlated.max()),
                    "paired_with": str(strongest_partner),
                }
            )
    correlated_to_drop = sorted(set(correlated_to_drop))
    correlation_filtered_columns = [column for column in variance_filtered.columns if column not in correlated_to_drop]
    correlation_filtered = variance_filtered[correlation_filtered_columns].copy()

    mutual_info_scores = mutual_info_classif(correlation_filtered, y_train, random_state=args.random_state)
    mutual_info_series = pd.Series(mutual_info_scores, index=correlation_filtered.columns).sort_values(ascending=False)
    selected_series = mutual_info_series[mutual_info_series >= args.mutual_info_threshold]
    fallback_used = False
    if selected_series.empty:
        fallback_used = True
        fallback_count = min(args.max_selected_features or len(mutual_info_series), max(32, min(128, len(mutual_info_series))))
        selected_series = mutual_info_series.head(fallback_count)

    if args.max_selected_features is not None and len(selected_series) > args.max_selected_features:
        selected_series = selected_series.sort_values(ascending=False).head(args.max_selected_features)

    selected_features = selected_series.index.tolist()
    for column, score in mutual_info_series.items():
        if column not in selected_features:
            removed_records.append({"feature": column, "reason": "low_mutual_information", "score": float(score)})

    details_frame = pd.DataFrame(removed_records)
    selection_report = {
        "initial_feature_count": int(X_train.shape[1]),
        "selected_feature_count": int(len(selected_features)),
        "removed_feature_count": int(X_train.shape[1] - len(selected_features)),
        "thresholds": {
            "missing_threshold": args.missing_threshold,
            "near_constant_threshold": args.near_constant_threshold,
            "correlation_threshold": args.correlation_threshold,
            "mutual_info_threshold": args.mutual_info_threshold,
            "max_selected_features": args.max_selected_features,
        },
        "fallback_used": fallback_used,
        "selected_features": [{"feature": feature, "mutual_information": float(selected_series[feature])} for feature in selected_features],
        "removed_features": details_frame.to_dict(orient="records"),
        "top_mutual_information_scores": {str(key): float(value) for key, value in selected_series.head(25).items()},
    }
    return selected_features, details_frame, selection_report


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_dataloaders(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, DataLoader]:
    def make_loader(features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    return {
        "train": make_loader(X_train, y_train, True),
        "val": make_loader(X_val, y_val, False),
        "test": make_loader(X_test, y_test, False),
    }


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(np.argmax(probs, axis=1))
    return np.concatenate(predictions), np.concatenate(probabilities)


def class_weights_for_mode(num_classes: int, y: np.ndarray, mode: str) -> np.ndarray:
    weights = np.ones(num_classes, dtype=np.float32)
    if mode == "none":
        return weights

    present_classes = np.unique(y)
    if len(present_classes) == 0:
        return weights

    computed = compute_class_weight(class_weight="balanced", classes=present_classes, y=y)
    for class_index, weight in zip(present_classes, computed):
        weights[int(class_index)] = float(weight)
    return weights


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    total_rows = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return running_loss / max(total_rows, 1)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    y_true_batches: List[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size
            y_true_batches.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true_batches)
    y_pred, probabilities = predict_probabilities(model, loader, device)
    metrics, report_df, confusion = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(total_rows, 1)
    return {
        "metrics": metrics,
        "report_df": report_df,
        "confusion": confusion,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
    }


def train_ft_transformer_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
    args: FTTrainingArgs,
    device: torch.device,
    logger,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, object], int, Optional[int]]:
    model = FTTransformerClassifier(
        n_features=X_train.shape[1],
        num_classes=len(class_names),
        d_token=args.hidden_dim,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    class_weight_values = class_weights_for_mode(len(class_names), y_train, args.class_weight_mode)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    dataloaders = build_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_val,
        y_test=y_val,
        batch_size=args.batch_size,
        device=device,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_metrics: Optional[Dict[str, object]] = None
    best_epoch = 1
    early_stop_epoch: Optional[int] = None
    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, dataloaders["train"], optimizer, criterion, device)
        val_eval = evaluate_loader(model, dataloaders["val"], criterion, device, class_names)
        val_metrics = val_eval["metrics"]
        val_macro_f1 = float(val_metrics["macro_f1"])
        selection_score = float(val_metrics[args.selection_metric])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_selection_score": selection_score,
            }
        )
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f | val_%s=%.4f",
            epoch,
            args.epochs,
            train_loss,
            float(val_metrics["loss"]),
            val_macro_f1,
            args.selection_metric,
            selection_score,
        )

        if scheduler is not None:
            scheduler.step(selection_score)

        if selection_score > best_score:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            early_stop_epoch = epoch
            logger.info("Early stopping triggered after %s epochs without improvement.", epochs_without_improvement)
            break

    model.load_state_dict(best_state)
    if best_metrics is None:
        best_eval = evaluate_loader(model, dataloaders["val"], criterion, device, class_names)
        best_metrics = best_eval["metrics"]
    return model, history, best_metrics, best_epoch, early_stop_epoch


def bundle_from_outputs(
    model_name: str,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray],
    class_names: Sequence[str],
    metrics_override: Optional[Dict[str, object]] = None,
) -> ModelEvaluationBundle:
    metrics, report_df, confusion = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
    )
    if metrics_override:
        metrics.update(metrics_override)
    return ModelEvaluationBundle(
        model_name=model_name,
        metrics=metrics,
        report_df=report_df,
        confusion=confusion,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )


def benchmark_models(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    random_state: int,
    logger,
) -> Tuple[pd.DataFrame, Dict[str, ModelEvaluationBundle], Optional[str], Optional[pd.Series]]:
    benchmark_rows: List[Dict[str, object]] = []
    model_evaluations: Dict[str, ModelEvaluationBundle] = {}
    feature_importance_series: Optional[pd.Series] = None
    feature_importance_source: Optional[str] = None
    fit_X = np.vstack([X_train, X_val])
    fit_y = np.concatenate([y_train, y_val])

    models: List[Tuple[str, object]] = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        xgb_params: Dict[str, object] = {
            "n_estimators": 320,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.5,
            "min_child_weight": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
            "missing": np.nan,
        }
        if len(class_names) > 2:
            xgb_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": len(class_names)})
        else:
            xgb_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        models.insert(0, ("XGBoost", XGBClassifier(**xgb_params)))
    except ImportError:
        logger.warning("xgboost is not installed; benchmark will skip the XGBoost baseline.")

    for model_name, model in models:
        logger.info("Training benchmark model: %s", model_name)
        model.fit(fit_X, fit_y)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        predictions = model.predict(X_test)
        bundle = bundle_from_outputs(
            model_name,
            y_true=y_test,
            y_pred=predictions,
            probabilities=probabilities,
            class_names=class_names,
        )
        model_evaluations[model_name] = bundle

        row = {"model": model_name}
        row.update({key: value for key, value in bundle.metrics.items() if key != "per_class"})
        benchmark_rows.append(row)

        if hasattr(model, "feature_importances_"):
            importance_candidate = pd.Series(model.feature_importances_)
            if feature_importance_source is None or model_name == "XGBoost":
                feature_importance_source = model_name
                feature_importance_series = importance_candidate

    return pd.DataFrame(benchmark_rows), model_evaluations, feature_importance_source, feature_importance_series


def model_slug(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(" ", "_")


def binary_probability_matrix(probabilities: Optional[np.ndarray], normal_index: int) -> Optional[np.ndarray]:
    if probabilities is None:
        return None
    normal_probabilities = np.clip(probabilities[:, normal_index], 0.0, 1.0)
    disease_probabilities = np.clip(1.0 - normal_probabilities, 0.0, 1.0)
    return np.column_stack([normal_probabilities, disease_probabilities])


def build_binary_bundle(bundle: ModelEvaluationBundle, normal_index: int) -> ModelEvaluationBundle:
    binary_classes = ["Normal", "Disease"]
    y_true_binary = (bundle.y_true != normal_index).astype(int)
    y_pred_binary = (bundle.y_pred != normal_index).astype(int)
    binary_probabilities = binary_probability_matrix(bundle.probabilities, normal_index)
    return bundle_from_outputs(
        bundle.model_name,
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        probabilities=binary_probabilities,
        class_names=binary_classes,
    )


def combine_report_frames(evaluations: Dict[str, ModelEvaluationBundle]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model_name, bundle in evaluations.items():
        frame = bundle.report_df.copy()
        frame.insert(0, "model", model_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def selected_feature_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    stats = dataframe.describe().transpose()[["mean", "std", "min", "max"]]
    stats["skewness"] = dataframe.skew(numeric_only=True)
    stats["missing_rate"] = dataframe.isna().mean()
    return stats.reset_index().rename(columns={"index": "feature"}).sort_values("feature").reset_index(drop=True)


def metrics_row(model_name: str, metrics: Dict[str, object]) -> Dict[str, object]:
    row = {"model": model_name}
    for key, value in metrics.items():
        if key == "per_class":
            continue
        row[key] = value
    return row


def main() -> None:
    args = parse_args()
    logger = setup_logging("train_ft_transformer.log")
    ensure_directories()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    layout = ensure_reporting_layout()
    paths = artifact_paths(layout)
    set_random_seed(args.random_state)

    config = build_shared_training_config(
        image_root=args.image_root,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        feature_set_mode=args.feature_set_mode,
        max_rows_per_dataset=args.max_rows_per_dataset,
        use_existing_split=args.use_existing_split,
    )
    combined = load_combined_gold_data(config=config, logger=logger)
    split_frames = split_feature_matrices(combined)
    split_target = split_labels(combined)

    selected_features, selection_details, selection_report = fit_feature_selector(
        X_train=split_frames["train"],
        y_train=split_target["train"],
        args=args,
    )
    if not selected_features:
        raise RuntimeError("Feature selection removed every feature. Lower the thresholds and try again.")

    split_frames = {split_name: frame[selected_features].copy() for split_name, frame in split_frames.items()}
    selected_feature_frame_all = combined.numeric_features[selected_features].copy()
    selected_feature_stats = selected_feature_statistics(selected_feature_frame_all)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(split_frames["train"]).astype(np.float32)
    X_val = preprocessor.transform(split_frames["val"]).astype(np.float32)
    X_test = preprocessor.transform(split_frames["test"]).astype(np.float32)
    y_train = split_target["train"]
    y_val = split_target["val"]
    y_test = split_target["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ft_model, history, val_metrics, best_epoch, early_stop_epoch = train_ft_transformer_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_names=combined.class_names,
        args=args,
        device=device,
        logger=logger,
    )

    test_loader = build_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=args.batch_size,
        device=device,
    )["test"]
    class_weight_values = class_weights_for_mode(len(combined.class_names), y_train, args.class_weight_mode)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device))
    test_eval = evaluate_loader(ft_model, test_loader, criterion, device, combined.class_names)
    test_metrics = test_eval["metrics"]
    test_metrics["loss"] = float(test_metrics["loss"])
    ft_bundle = ModelEvaluationBundle(
        model_name="FT-Transformer",
        metrics=test_metrics,
        report_df=test_eval["report_df"],
        confusion=test_eval["confusion"],
        y_true=test_eval["y_true"],
        y_pred=test_eval["y_pred"],
        probabilities=test_eval["probabilities"],
    )

    benchmark_results, benchmark_evaluations, feature_importance_source, feature_importance_values = benchmark_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=combined.class_names,
        random_state=args.random_state,
        logger=logger,
    )

    all_evaluations = {"FT-Transformer": ft_bundle}
    all_evaluations.update(benchmark_evaluations)

    ft_row = metrics_row("FT-Transformer", test_metrics)
    benchmark_results = pd.concat([benchmark_results, pd.DataFrame([ft_row])], ignore_index=True)
    benchmark_results["rank_macro_f1"] = benchmark_results["macro_f1"].rank(ascending=False, method="min")
    benchmark_results = benchmark_results.sort_values(["rank_macro_f1", "model"]).reset_index(drop=True)

    normal_index = combined.label_to_index.get("Normal")
    if normal_index is None:
        raise RuntimeError("Binary analysis requires a 'Normal' class label.")

    binary_label_names = np.where(combined.dataframe["label"].astype(str).to_numpy() == "Normal", "Normal", "Disease")
    binary_distribution = pd.Series(binary_label_names).value_counts().reindex(["Normal", "Disease"]).fillna(0).astype(int)
    binary_distribution_summary = pd.DataFrame(
        {
            "label": binary_distribution.index.astype(str),
            "count": binary_distribution.to_numpy(dtype=int),
            "percentage": (binary_distribution.to_numpy(dtype=float) / float(binary_distribution.sum())) * 100.0,
        }
    )

    binary_evaluations = {model_name: build_binary_bundle(bundle, normal_index) for model_name, bundle in all_evaluations.items()}
    binary_metrics_frame = pd.DataFrame(
        [metrics_row(model_name, bundle.metrics) for model_name, bundle in binary_evaluations.items()]
    ).sort_values(["macro_f1", "model"], ascending=[False, True]).reset_index(drop=True)
    binary_per_class_frame = combine_report_frames(binary_evaluations)
    multiclass_per_class_frame = combine_report_frames(all_evaluations)

    benchmark_results.to_csv(paths["benchmark_results"], index=False)
    save_table(benchmark_results, paths["per_model_metrics"])
    save_table(multiclass_per_class_frame, paths["per_class_metrics"])
    save_table(binary_distribution_summary, paths["binary_label_distribution_summary"])
    save_table(binary_metrics_frame, paths["binary_metrics_summary"])
    save_table(binary_per_class_frame, paths["binary_per_class_metrics"])
    save_table(selected_feature_stats, paths["selected_feature_statistics"])

    torch.save(
        {
            "model_state_dict": ft_model.state_dict(),
            "model_config": {
                "n_features": int(X_train.shape[1]),
                "num_classes": int(len(combined.class_names)),
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
        },
        paths["model"],
    )
    write_json(paths["label_mapping"], {"classes": combined.class_names, "label_to_index": combined.label_to_index})
    write_json(paths["selected_features"], {"selected_features": selected_features})
    write_json(
        paths["preprocessing_config"],
        {
            "model_type": "FT-Transformer",
            "image_root": str(args.image_root) if args.image_root else None,
            "feature_set_mode": args.feature_set_mode,
            "selected_features_artifact": str(paths["selected_features"]),
            "label_mapping_artifact": str(paths["label_mapping"]),
            "preprocessor_artifact": str(paths["preprocessor"]),
            "missing_value_strategy": "median_imputation",
            "scaling": "standard_scaler",
            "split_info": combined.split_info,
            "feature_selection_thresholds": selection_report["thresholds"],
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "patience": args.patience,
                "scheduler": args.scheduler,
                "selection_metric": args.selection_metric,
                "class_weight_mode": args.class_weight_mode,
            },
            "reporting_artifacts": {
                "plots_root": str(paths["plots_root"]),
                "tables_root": str(paths["tables_root"]),
            },
        },
    )
    joblib.dump(preprocessor, paths["preprocessor"])
    write_json(paths["feature_selection_report"], selection_report)
    selection_details.to_csv(paths["feature_selection_details"], index=False)
    ft_bundle.report_df.to_csv(paths["classification_report"], index=False)

    history_frame = pd.DataFrame(history)
    save_table(history_frame, paths["training_history_csv"])
    training_plot = layout["training"] / "ft_transformer_training_validation_performance.png"
    plot_training_history(
        history_frame,
        output_paths=[training_plot, paths["training_history"]],
        title=f"FT-Transformer Training and Validation Performance ({args.epochs}-Epoch Run)",
        subtitle="Train loss, validation loss, and validation Macro F1 with the best checkpoint and early stopping behavior highlighted.",
        best_epoch=best_epoch,
        early_stop_epoch=early_stop_epoch,
    )

    if feature_importance_values is not None and feature_importance_source is not None:
        feature_importance_frame = (
            pd.DataFrame(
                {
                    "feature": selected_features,
                    "importance": feature_importance_values.to_numpy(dtype=float),
                    "source_model": feature_importance_source,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        save_table(feature_importance_frame, paths["feature_importance_values"])
        feature_importance_plot = layout["feature_analysis"] / "top_benchmark_model_feature_importances.png"
        plot_ranked_horizontal_bars(
            feature_importance_frame.head(20).set_index("feature")["importance"],
            output_paths=[feature_importance_plot, paths["feature_importance_plot"]],
            title=f"Top Features Ranked by {feature_importance_source} Importance",
            subtitle=f"The benchmark feature-importance profile shown here is generated by {feature_importance_source}.",
            xlabel="Importance",
            color=MODEL_COLORS.get(feature_importance_source, BRAND_COLORS["plum"]),
            value_fmt="{:.4f}",
        )

    multiclass_confusion_paths: List[tuple[Path, str]] = []
    for model_name, bundle in all_evaluations.items():
        slug = model_slug(model_name)
        raw_path = layout["confusion_matrices"] / f"cm_{slug}_raw.png"
        normalized_path = layout["confusion_matrices"] / f"cm_{slug}_normalized.png"
        raw_outputs: List[Path] = [raw_path]
        if model_name == "FT-Transformer":
            raw_outputs.append(paths["confusion_matrix"])
        plot_confusion_matrix(
            bundle.confusion,
            class_names=combined.class_names,
            output_paths=raw_outputs,
            title=f"Confusion Matrix - {model_name}",
            subtitle="Raw prediction counts across the full multi-class test set.",
            normalize=False,
        )
        plot_confusion_matrix(
            bundle.confusion,
            class_names=combined.class_names,
            output_paths=normalized_path,
            title=f"Normalized Confusion Matrix - {model_name}",
            subtitle="Row-normalized view of per-class recall and confusion patterns.",
            normalize=True,
        )
        multiclass_confusion_paths.append((raw_path, f"Raw multi-class confusion matrix for {model_name}."))
        multiclass_confusion_paths.append((normalized_path, f"Normalized multi-class confusion matrix for {model_name}."))

    binary_confusion_paths: List[tuple[Path, str]] = []
    for model_name, bundle in binary_evaluations.items():
        slug = model_slug(model_name)
        raw_path = layout["binary"] / f"binary_cm_{slug}_raw.png"
        normalized_path = layout["binary"] / f"binary_cm_{slug}_normalized.png"
        plot_confusion_matrix(
            bundle.confusion,
            class_names=["Normal", "Disease"],
            output_paths=raw_path,
            title=f"Binary Confusion Matrix - {model_name}",
            subtitle="Raw prediction counts after collapsing classes into Normal versus Disease.",
            normalize=False,
        )
        plot_confusion_matrix(
            bundle.confusion,
            class_names=["Normal", "Disease"],
            output_paths=normalized_path,
            title=f"Normalized Binary Confusion Matrix - {model_name}",
            subtitle="Row-normalized binary confusion matrix for Normal versus Disease.",
            normalize=True,
        )
        binary_confusion_paths.append((raw_path, f"Raw binary confusion matrix for {model_name} after collapsing disease labels into a single class."))
        binary_confusion_paths.append((normalized_path, f"Normalized binary confusion matrix for {model_name} in the Normal versus Disease setting."))

    benchmark_plot = layout["benchmarks"] / "benchmark_model_metrics_comparison.png"
    plot_grouped_metric_bars(
        benchmark_results.sort_values("rank_macro_f1"),
        metric_columns=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
        output_paths=benchmark_plot,
        title="Comparison of Benchmark Models on Multi-Class Classification",
        subtitle="FT-Transformer is highlighted against classical baselines using both raw and balance-sensitive metrics.",
    )

    macro_f1_plot = layout["benchmarks"] / "macro_f1_comparison_across_models.png"
    plot_metric_highlight(
        benchmark_results,
        metric_column="macro_f1",
        output_paths=macro_f1_plot,
        title="Macro F1 Comparison Across Models",
        subtitle="Macro F1 is emphasized because it better reflects performance under severe class imbalance.",
    )

    tradeoff_plot = layout["benchmarks"] / "accuracy_vs_macro_f1_tradeoff.png"
    plot_metric_tradeoff(
        benchmark_results,
        x_column="accuracy",
        y_column="macro_f1",
        output_paths=tradeoff_plot,
        title="Accuracy vs Macro F1 Trade-off Across Benchmark Models",
        subtitle="This view highlights why FT-Transformer is preferred for balance-sensitive evaluation even when some baselines reach higher raw accuracy.",
    )

    binary_distribution_plot = layout["binary"] / "binary_class_distribution_normal_vs_disease.png"
    plot_categorical_distribution(
        binary_distribution_summary,
        label_column="label",
        count_column="count",
        output_paths=binary_distribution_plot,
        title="Binary Class Distribution (Normal vs Disease)",
        subtitle="Counts are shown above each bar after mapping Normal to 0 and every disease label to 1.",
        ylabel="Image Count",
        color=BRAND_COLORS["teal"],
        rotate_xticks=0,
    )

    binary_benchmark_plot = layout["binary"] / "binary_benchmark_summary.png"
    plot_grouped_metric_bars(
        binary_metrics_frame,
        metric_columns=["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"],
        output_paths=binary_benchmark_plot,
        title="Binary Benchmark Comparison (Normal vs Disease)",
        subtitle="Binary evaluation is derived by collapsing multi-class predictions into Normal versus Disease.",
    )

    binary_curve_records: List[Dict[str, object]] = []
    for model_name, bundle in binary_evaluations.items():
        if bundle.probabilities is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(bundle.y_true, bundle.probabilities[:, 1])
        except ValueError:
            continue
        binary_curve_records.append(
            {
                "model": model_name,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": float(bundle.metrics.get("roc_auc", 0.0) or 0.0),
            }
        )
    binary_roc_plot = layout["binary"] / "binary_roc_curves_all_models.png"
    if binary_curve_records:
        plot_binary_roc_comparison(
            binary_curve_records,
            output_paths=binary_roc_plot,
            title="Binary ROC Curves (Normal vs Disease)",
            subtitle="Disease probability is computed as one minus the Normal-class probability from the original multi-class models.",
        )

    multiclass_roc_plot = layout["evaluation"] / "ft_transformer_multiclass_ovr_roc_curves.png"
    multiclass_roc_summary = plot_multiclass_ovr_curves(
        y_true=ft_bundle.y_true,
        probabilities=ft_bundle.probabilities,
        class_names=combined.class_names,
        model_name="FT-Transformer",
        output_paths=multiclass_roc_plot,
        title="FT-Transformer Multi-Class One-vs-Rest ROC Curves",
        subtitle="Per-class ROC curves for the primary model on the held-out test split.",
        curve_type="roc",
    )
    if not multiclass_roc_summary.empty:
        save_table(multiclass_roc_summary, paths["multiclass_ovr_roc_summary"])

    multiclass_pr_plot = layout["evaluation"] / "ft_transformer_multiclass_precision_recall_curves.png"
    multiclass_pr_summary = plot_multiclass_ovr_curves(
        y_true=ft_bundle.y_true,
        probabilities=ft_bundle.probabilities,
        class_names=combined.class_names,
        model_name="FT-Transformer",
        output_paths=multiclass_pr_plot,
        title="FT-Transformer Multi-Class Precision-Recall Curves",
        subtitle="Per-class precision-recall curves provide a paper-ready view under label imbalance.",
        curve_type="pr",
    )
    if not multiclass_pr_summary.empty:
        save_table(multiclass_pr_summary, paths["multiclass_ovr_pr_summary"])

    per_class_recall_plot = layout["evaluation"] / "ft_transformer_per_class_recall.png"
    plot_per_class_recall(
        ft_bundle.report_df,
        model_name="FT-Transformer",
        output_paths=per_class_recall_plot,
        title="FT-Transformer Per-Class Recall on the Multi-Class Test Set",
        subtitle="Per-class recall highlights the classes that remain hardest to recover under imbalance.",
    )

    misclassified_summary = summarize_top_misclassifications(ft_bundle.confusion, combined.class_names)
    save_table(misclassified_summary, paths["top_misclassified_classes"])

    feature_count_plot = layout["feature_analysis"] / "feature_selection_before_vs_after_comparison.png"
    plot_feature_selection_counts(
        before_count=len(combined.feature_columns),
        after_count=len(selected_features),
        output_paths=feature_count_plot,
        title="Feature Selection Before-vs-After Comparison",
        subtitle="This figure compares the merged Gold-layer feature count before final selection versus the final 400-feature FT-Transformer input space.",
    )

    figure_descriptions: List[tuple[Path, str]] = [
        (training_plot, "FT-Transformer training and validation curves with the best epoch and early stopping behavior highlighted."),
        (benchmark_plot, "Grouped benchmark comparison across accuracy, balanced accuracy, Macro F1, and weighted F1."),
        (macro_f1_plot, "Macro F1 ranking across FT-Transformer and benchmark baselines."),
        (tradeoff_plot, "Accuracy versus Macro F1 trade-off plot showing why balance-sensitive metrics matter for model selection."),
        (binary_distribution_plot, "Binary class distribution after collapsing all disease labels into a single positive class."),
        (binary_benchmark_plot, "Binary benchmark comparison derived from multi-class model outputs."),
        (feature_count_plot, "Feature-count comparison before and after the final 400-feature selection stage."),
        (multiclass_roc_plot, "FT-Transformer one-vs-rest ROC curves across all disease classes."),
        (multiclass_pr_plot, "FT-Transformer one-vs-rest precision-recall curves across all disease classes."),
        (per_class_recall_plot, "Per-class recall profile for the FT-Transformer on the held-out multi-class test set."),
    ]
    if feature_importance_values is not None and feature_importance_source is not None:
        figure_descriptions.append(
            (
                layout["feature_analysis"] / "top_benchmark_model_feature_importances.png",
                f"Top selected features ranked by {feature_importance_source} importance.",
            )
        )
    if binary_curve_records:
        figure_descriptions.append((binary_roc_plot, "Binary ROC comparison across all models after collapsing predictions to Normal versus Disease."))
    figure_descriptions.extend(multiclass_confusion_paths)
    figure_descriptions.extend(binary_confusion_paths)
    append_figure_descriptions(figure_descriptions, paths["figure_descriptions"])

    best_baseline = benchmark_results[benchmark_results["model"] != "FT-Transformer"].sort_values("macro_f1", ascending=False).head(1)
    ft_outperformed_baselines = True
    if not best_baseline.empty:
        ft_outperformed_baselines = bool(float(test_metrics["macro_f1"]) > float(best_baseline.iloc[0]["macro_f1"]))

    final_metrics = {
        "final_model": "FT-Transformer",
        "device": str(device),
        "selected_feature_count": int(len(selected_features)),
        "training_strategy": {
            "selection_metric": args.selection_metric,
            "class_weight_mode": args.class_weight_mode,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "benchmark_summary": {
            "results": benchmark_results.to_dict(orient="records"),
            "ft_transformer_outperformed_all_baselines_by_macro_f1": ft_outperformed_baselines,
        },
        "binary_summary": {
            "mapping": {"Normal": 0, "Disease": 1},
            "label_distribution": {row["label"]: int(row["count"]) for _, row in binary_distribution_summary.iterrows()},
            "results": binary_metrics_frame.to_dict(orient="records"),
        },
        "data_summary": {
            "rows_total": int(len(combined.dataframe)),
            "feature_columns_before_selection": int(len(combined.feature_columns)),
            "feature_columns_after_selection": int(len(selected_features)),
            "split_counts": {split_name: int(mask.sum()) for split_name, mask in combined.split_masks.items()},
            "dataset_sources": combined.dataset_summaries,
            "dropped_rows": int(combined.dropped_rows),
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["final_metrics"], final_metrics)

    summary_lines = [
        "FT-Transformer Final Summary",
        f"Device: {device}",
        f"Classes: {len(combined.class_names)}",
        f"Features before selection: {len(combined.feature_columns)}",
        f"Configured max selected features: {args.max_selected_features}",
        f"Features after selection: {len(selected_features)}",
        f"Checkpoint selection metric: {args.selection_metric}",
        f"Class weight mode: {args.class_weight_mode}",
        f"Best validation macro F1: {float(val_metrics['macro_f1']):.4f}",
        f"Best validation {args.selection_metric}: {float(val_metrics[args.selection_metric]):.4f}",
        f"Test accuracy: {float(test_metrics['accuracy']):.4f}",
        f"Test balanced accuracy: {float(test_metrics['balanced_accuracy']):.4f}",
        f"Test macro F1: {float(test_metrics['macro_f1']):.4f}",
        f"Test weighted F1: {float(test_metrics['weighted_f1']):.4f}",
        f"Binary counts - Normal: {int(binary_distribution.get('Normal', 0))}",
        f"Binary counts - Disease: {int(binary_distribution.get('Disease', 0))}",
        f"Plots saved to: {paths['plots_root']}",
        f"Tables saved to: {paths['tables_root']}",
        f"Saved model: {paths['model']}",
    ]
    write_text(paths["metrics_summary"], "\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
