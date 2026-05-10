from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import MODELS_ROOT, PROJECT_ROOT as COMMON_PROJECT_ROOT, read_dataframe, resolve_existing_table_path  # noqa: E402
from scripts.feature_extractor import ViTLargeFeatureExtractor  # noqa: E402

DEFAULT_FEATURE_EXTRACTOR_MODEL = "google/vit-large-patch16-224-in21k"


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


@dataclass(frozen=True)
class FTInferenceRuntime:
    model: FTTransformerClassifier
    class_names: List[str]
    selected_features: List[str]
    preprocessing_config: Dict[str, Any]
    preprocessor: Any
    paths: Dict[str, Path]
    feature_extractor_model_name: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_artifact_paths() -> Dict[str, Path]:
    paths = {
        "model": MODELS_ROOT / "best_ft_transformer.pth",
        "label_mapping": MODELS_ROOT / "label_mapping.json",
        "selected_features": MODELS_ROOT / "selected_features.json",
        "preprocessing_config": MODELS_ROOT / "preprocessing_config.json",
        "preprocessor": MODELS_ROOT / "ft_preprocessor.joblib",
    }
    return {
        "model": paths["model"],
        "label_mapping": paths["label_mapping"],
        "selected_features": paths["selected_features"],
        "preprocessing_config": paths["preprocessing_config"],
        "preprocessor": paths["preprocessor"],
    }


def missing_artifacts() -> List[str]:
    return [str(path) for path in required_artifact_paths().values() if not path.exists()]


def confidence_band(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.55:
        return "Moderate"
    return "Low"


def confidence_note(confidence: float) -> str:
    if confidence >= 0.80:
        return "Model confidence is high."
    if confidence >= 0.55:
        return "Model confidence is moderate."
    return "Prediction uncertainty is elevated. Review the ranked alternatives."


def resolve_feature_extractor_model_name(preprocessing_config: Optional[Dict[str, Any]] = None) -> str:
    if preprocessing_config and preprocessing_config.get("feature_extractor_model_name"):
        return str(preprocessing_config["feature_extractor_model_name"])

    legacy_inference_config = MODELS_ROOT / "inference_config.json"
    if legacy_inference_config.exists():
        payload = load_json(legacy_inference_config)
        if payload.get("feature_extractor_model_name"):
            return str(payload["feature_extractor_model_name"])

    return DEFAULT_FEATURE_EXTRACTOR_MODEL


@lru_cache(maxsize=1)
def load_runtime_bundle() -> FTInferenceRuntime:
    missing = missing_artifacts()
    if missing:
        raise FileNotFoundError(
            "Required FT-Transformer artifacts are missing. Run `python scripts/train_ft_transformer.py` first.\n"
            + "\n".join(missing)
        )

    paths = required_artifact_paths()
    label_mapping = load_json(paths["label_mapping"])
    selected_features_payload = load_json(paths["selected_features"])
    preprocessing_config = load_json(paths["preprocessing_config"])
    preprocessor = joblib.load(paths["preprocessor"])
    feature_extractor_model_name = resolve_feature_extractor_model_name(preprocessing_config)

    try:
        checkpoint = torch.load(paths["model"], map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(paths["model"], map_location="cpu")
    model_config = checkpoint["model_config"]
    model = FTTransformerClassifier(
        n_features=int(model_config["n_features"]),
        num_classes=int(model_config["num_classes"]),
        d_token=int(model_config["hidden_dim"]),
        n_heads=int(model_config["num_heads"]),
        n_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return FTInferenceRuntime(
        model=model,
        class_names=list(label_mapping["classes"]),
        selected_features=list(selected_features_payload["selected_features"]),
        preprocessing_config=preprocessing_config,
        preprocessor=preprocessor,
        paths=paths,
        feature_extractor_model_name=feature_extractor_model_name,
    )


@lru_cache(maxsize=1)
def load_feature_extractor() -> ViTLargeFeatureExtractor:
    runtime = load_runtime_bundle()
    return ViTLargeFeatureExtractor(model_name=runtime.feature_extractor_model_name)


def available_gold_files() -> Dict[str, Path]:
    gold_dir = COMMON_PROJECT_ROOT / "gold"
    choices: Dict[str, Path] = {}
    for path in sorted(gold_dir.glob("gold_*.*")):
        if path.suffix.lower() not in {".csv", ".xlsx"}:
            continue
        label = path.name
        choices[label] = path
    return choices


def load_input_frame(input_path: Path, row_index: Optional[int] = None) -> pd.DataFrame:
    dataframe = read_dataframe(resolve_existing_table_path(Path(input_path)))
    if row_index is not None:
        if row_index < 0 or row_index >= len(dataframe):
            raise IndexError(f"Row index {row_index} is out of range for {input_path.name}.")
        dataframe = dataframe.iloc[[row_index]].copy()
    return dataframe


def example_manual_payload() -> str:
    gold_files = list(available_gold_files().values())
    if not gold_files:
        runtime = load_runtime_bundle()
        return json.dumps({feature: 0.0 for feature in runtime.selected_features[:12]}, indent=2)

    example_frame = load_input_frame(gold_files[0], row_index=0)
    sample = example_frame.iloc[0].to_dict()
    compact_sample = {key: sample[key] for key in list(sample)[: min(18, len(sample))]}
    return json.dumps(compact_sample, indent=2, default=str)


def prepare_model_frame(input_frame: pd.DataFrame, selected_features: Sequence[str]) -> pd.DataFrame:
    records = [
        {column: row.get(column, np.nan) for column in selected_features}
        for row in input_frame.to_dict(orient="records")
    ]
    model_frame = pd.DataFrame(records, columns=list(selected_features))
    return model_frame.apply(pd.to_numeric, errors="coerce")


def metadata_columns_for_frame(input_frame: pd.DataFrame) -> List[str]:
    return [column for column in input_frame.columns if not str(column).startswith("feature_")]


def predict_probabilities(model_frame: pd.DataFrame) -> np.ndarray:
    runtime = load_runtime_bundle()
    transformed = runtime.preprocessor.transform(model_frame).astype(np.float32)

    with torch.no_grad():
        logits = runtime.model(torch.tensor(transformed, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


def build_prediction_results(
    *,
    probabilities: np.ndarray,
    metadata_rows: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    runtime = load_runtime_bundle()

    results: List[Dict[str, Any]] = []
    for row_index, row_probabilities in enumerate(probabilities):
        predicted_index = int(np.argmax(row_probabilities))
        confidence = float(row_probabilities[predicted_index])
        ordered_indices = np.argsort(row_probabilities)[::-1][: max(1, min(top_k, len(runtime.class_names)))]
        top_predictions = [
            {
                "rank": position + 1,
                "class": runtime.class_names[class_index],
                "probability": float(row_probabilities[class_index]),
            }
            for position, class_index in enumerate(ordered_indices)
        ]
        results.append(
            {
                "predicted_label": runtime.class_names[predicted_index],
                "confidence": confidence,
                "confidence_band": confidence_band(confidence),
                "confidence_note": confidence_note(confidence),
                "top_predictions": top_predictions,
                "metadata": metadata_rows[row_index],
            }
        )
    return results


def predict_from_frame(input_frame: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    runtime = load_runtime_bundle()
    model_frame = prepare_model_frame(input_frame, runtime.selected_features)
    probabilities = predict_probabilities(model_frame)
    metadata_columns = metadata_columns_for_frame(input_frame)
    metadata_rows = [
        {
            str(column): input_frame.iloc[row_index][column]
            for column in metadata_columns
            if column in input_frame.columns
        }
        for row_index in range(len(input_frame))
    ]
    return build_prediction_results(probabilities=probabilities, metadata_rows=metadata_rows, top_k=top_k)


def feature_snapshot_from_mapping(feature_mapping: Dict[str, float], limit: int = 12) -> pd.DataFrame:
    runtime = load_runtime_bundle()
    rows = []
    for feature_name in runtime.selected_features[:limit]:
        rows.append({"Feature": feature_name, "Value": float(feature_mapping.get(feature_name, np.nan))})
    return pd.DataFrame(rows)


def predict_from_pil_image(image: Image.Image, top_k: int = 3, image_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    runtime = load_runtime_bundle()
    if image is None:
        raise ValueError("No image was provided for inference.")

    original_size = image.size
    original_mode = image.mode
    feature_extractor = load_feature_extractor()
    rgb_image = image.convert("RGB")
    embedding = feature_extractor.extract_single_image(rgb_image)
    feature_mapping = {f"feature_{index:04d}": float(value) for index, value in enumerate(embedding)}
    missing_selected = [feature for feature in runtime.selected_features if feature not in feature_mapping]
    if missing_selected:
        raise ValueError(
            "The extracted ViT embedding does not cover every selected FT-Transformer feature. "
            f"Missing examples: {missing_selected[:10]}"
        )

    model_frame = pd.DataFrame(
        [{feature: feature_mapping[feature] for feature in runtime.selected_features}],
        columns=runtime.selected_features,
    )
    probabilities = predict_probabilities(model_frame)
    metadata_rows = [
        {
            "source": image_name or "uploaded_image",
            "image_width": int(original_size[0]),
            "image_height": int(original_size[1]),
            "image_mode": str(original_mode),
            "feature_extractor": runtime.feature_extractor_model_name,
        }
    ]
    results = build_prediction_results(probabilities=probabilities, metadata_rows=metadata_rows, top_k=top_k)
    if results:
        results[0]["gradcam"] = feature_extractor.build_attention_overlay(rgb_image)
        results[0]["gradcam_method"] = "ViT last-layer CLS attention overlay"
    return results, feature_snapshot_from_mapping(feature_mapping)


def model_info_snapshot() -> Dict[str, Any]:
    runtime = load_runtime_bundle()
    info: Dict[str, Any] = {
        "model_name": "ViT feature extractor + FT-Transformer classifier",
        "pipeline": {
            "feature_extraction_model": runtime.feature_extractor_model_name,
            "classification_model": "FT-Transformer",
        },
        "selected_feature_count": len(runtime.selected_features),
        "class_count": len(runtime.class_names),
        "classes": runtime.class_names,
        "feature_extractor_name": runtime.feature_extractor_model_name,
        "preprocessing": {
            "missing_value_strategy": runtime.preprocessing_config.get("missing_value_strategy"),
            "scaling": runtime.preprocessing_config.get("scaling"),
        },
        "artifact_paths": {key: str(value) for key, value in runtime.paths.items()},
    }
    final_metrics_path = COMMON_PROJECT_ROOT / "artifacts" / "reports" / "final_metrics.json"
    if final_metrics_path.exists():
        info["final_metrics"] = load_json(final_metrics_path)
    return info
