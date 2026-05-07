from __future__ import annotations

import argparse
import base64
from io import BytesIO
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    PipelineConfig,
    ensure_directories,
    resolve_image_path,
    select_dataset_artifacts,
    setup_logging,
)


class ViTLargeFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import AutoImageProcessor, ViTModel

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        if self.device.type == "cpu":
            cpu_threads = os.cpu_count() or torch.get_num_threads()
            torch.set_num_threads(max(torch.get_num_threads(), cpu_threads))
            torch.set_num_interop_threads(max(torch.get_num_interop_threads(), cpu_threads))
        cache_dir_value = str(cache_dir) if cache_dir else None
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir_value,
                use_fast=True,
                local_files_only=True,
            )
        except OSError:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir_value,
                use_fast=True,
            )

        try:
            try:
                self.model = ViTModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir_value,
                    local_files_only=True,
                    attn_implementation="eager",
                )
            except TypeError:
                self.model = ViTModel.from_pretrained(model_name, cache_dir=cache_dir_value, local_files_only=True)
        except OSError:
            try:
                self.model = ViTModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir_value,
                    attn_implementation="eager",
                )
            except TypeError:
                self.model = ViTModel.from_pretrained(model_name, cache_dir=cache_dir_value)
        if hasattr(self.model, "set_attn_implementation"):
            try:
                self.model.set_attn_implementation("eager")
            except Exception:
                pass
        self.model.to(self.device)
        self.model.eval()

    def extract_from_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        prepared_images = [image.convert("RGB") for image in images]
        if not prepared_images:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self.processor(images=prepared_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

        return embeddings

    def extract_single_image(self, image: Image.Image) -> np.ndarray:
        embeddings = self.extract_from_pil_images([image])
        if embeddings.size == 0:
            raise ValueError("No embedding was produced for the provided image.")
        return embeddings[0]

    def build_attention_overlay(self, image: Image.Image, alpha: float = 0.42) -> str:
        prepared_image = image.convert("RGB")
        original_size = prepared_image.size
        inputs = self.processor(images=[prepared_image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs, output_attentions=True)

        heatmap: np.ndarray
        attentions = getattr(outputs, "attentions", None)
        if attentions:
            last_attention = attentions[-1][0].detach().cpu()
            cls_attention = last_attention.mean(dim=0)[0, 1:].numpy()
            grid_size = int(np.sqrt(cls_attention.shape[0]))
            heatmap = cls_attention[: grid_size * grid_size].reshape(grid_size, grid_size)
        else:
            patch_embeddings = outputs.last_hidden_state[0, 1:, :].detach().cpu().numpy()
            patch_energy = np.linalg.norm(patch_embeddings, axis=1)
            grid_size = int(np.sqrt(patch_energy.shape[0]))
            heatmap = patch_energy[: grid_size * grid_size].reshape(grid_size, grid_size)

        heatmap = heatmap.astype(np.float32)
        heatmap -= float(heatmap.min())
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap /= max_value

        heatmap_image = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(original_size, Image.Resampling.BICUBIC)
        heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0
        source_array = np.asarray(prepared_image).astype(np.float32)

        color_array = np.zeros_like(source_array)
        color_array[..., 0] = 255.0
        color_array[..., 1] = 210.0 * heatmap_array
        color_array[..., 2] = 35.0 * (1.0 - heatmap_array)
        blend_weight = np.expand_dims(heatmap_array * alpha, axis=2)
        overlay_array = source_array * (1.0 - blend_weight) + color_array * blend_weight

        output = Image.fromarray(np.uint8(np.clip(overlay_array, 0, 255)), mode="RGB")
        buffer = BytesIO()
        output.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def extract_batch(self, image_paths: Sequence[Path]) -> Tuple[np.ndarray, List[int], List[Tuple[str, str]]]:
        images: List[Image.Image] = []
        successful_indices: List[int] = []
        failures: List[Tuple[str, str]] = []

        for index, image_path in enumerate(image_paths):
            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
                successful_indices.append(index)
            except Exception as exc:  # pragma: no cover - defensive path
                failures.append((str(image_path), str(exc)))

        if not images:
            return np.empty((0, 0), dtype=np.float32), successful_indices, failures

        embeddings = self.extract_from_pil_images(images)
        return embeddings, successful_indices, failures
def prepare_metadata(
    metadata_path: Path,
    image_root: Optional[Path],
    max_rows: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    df = pd.read_csv(metadata_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    required_columns = {"dataset", "path"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns in {metadata_path}: {sorted(missing_columns)}")

    resolved_paths = [
        resolve_image_path(raw_path=row.path, dataset_name=row.dataset, image_root=image_root)
        for row in df.itertuples(index=False)
    ]
    df = df.copy()
    df["_resolved_image_path"] = [str(path) if path else None for path in resolved_paths]

    total_rows = len(df)
    existing_rows = int(df["_resolved_image_path"].notna().sum())
    missing_rows = int(total_rows - existing_rows)
    missing_examples = df.loc[df["_resolved_image_path"].isna(), "path"].head(10).tolist()

    stats = {
        "rows_in_metadata": total_rows,
        "rows_with_images": existing_rows,
        "rows_missing_images": missing_rows,
    }
    return df, stats, missing_examples


def extract_features_for_dataset(
    metadata_path: Path,
    output_path: Path,
    config: PipelineConfig,
    dry_run: bool = False,
    extractor: Optional[ViTLargeFeatureExtractor] = None,
) -> Dict[str, object]:
    logger = setup_logging("feature_extractor.log")
    ensure_directories()

    logger.info("Preparing metadata for feature extraction: %s", metadata_path)
    prepared_df, stats, missing_examples = prepare_metadata(
        metadata_path=metadata_path,
        image_root=config.image_root,
        max_rows=config.max_rows,
    )

    logger.info("Feature extraction input stats: %s", stats)
    if missing_examples:
        logger.warning("Sample missing image paths: %s", missing_examples)

    if dry_run:
        logger.info("Dry run enabled. Skipping model loading and feature extraction for %s", metadata_path.name)
        return stats

    if stats["rows_with_images"] == 0:
        raise FileNotFoundError(
            "No images were found for "
            f"{metadata_path.name}. Set --image-root to the folder that contains the NIH/TCIA/RSNA image trees."
        )

    dataset_extractor = extractor or ViTLargeFeatureExtractor(
        model_name=config.model_name,
        cache_dir=config.model_cache_dir,
        device=config.device,
    )
    if extractor is None:
        logger.info("Loaded model %s on device %s", config.model_name, dataset_extractor.device)

    rows_to_process = prepared_df.loc[prepared_df["_resolved_image_path"].notna()].copy()
    rows_to_process.reset_index(drop=True, inplace=True)

    metadata_batches: List[pd.DataFrame] = []
    embedding_batches: List[pd.DataFrame] = []
    feature_columns: Optional[List[str]] = None
    processed_rows = 0
    failed_after_open = 0

    for batch_start in range(0, len(rows_to_process), config.batch_size):
        batch_df = rows_to_process.iloc[batch_start : batch_start + config.batch_size].copy()
        image_paths = [Path(path_value) for path_value in batch_df["_resolved_image_path"].tolist()]
        embeddings, successful_indices, failures = dataset_extractor.extract_batch(image_paths)

        if failures:
            failed_after_open += len(failures)
            logger.warning("Failed to open %s image(s) in current batch", len(failures))

        if embeddings.size == 0:
            continue

        if feature_columns is None:
            feature_columns = [f"{config.feature_prefix}{index:04d}" for index in range(embeddings.shape[1])]

        successful_metadata = batch_df.iloc[successful_indices].drop(columns=["_resolved_image_path"]).reset_index(drop=True)
        successful_embeddings = pd.DataFrame(embeddings, columns=feature_columns)

        metadata_batches.append(successful_metadata)
        embedding_batches.append(successful_embeddings)
        processed_rows += len(successful_metadata)

        current_batch = (batch_start // config.batch_size) + 1
        if current_batch == 1 or current_batch % 25 == 0 or batch_start + config.batch_size >= len(rows_to_process):
            logger.info(
                "Processed %s/%s rows for %s",
                processed_rows,
                len(rows_to_process),
                metadata_path.name,
            )

    if not metadata_batches or feature_columns is None:
        raise RuntimeError(f"No embeddings were produced for {metadata_path.name}")

    metadata_df = pd.concat(metadata_batches, ignore_index=True)
    embeddings_df = pd.concat(embedding_batches, ignore_index=True)
    silver_df = pd.concat([metadata_df, embeddings_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver_df.to_excel(output_path, index=False, engine="openpyxl")

    result = {
        **stats,
        "rows_with_embeddings": int(len(silver_df)),
        "rows_failed_during_open": int(failed_after_open),
        "feature_count": int(len(feature_columns)),
        "output_path": str(output_path),
    }
    logger.info("Saved %s rows with %s features to %s", len(silver_df), len(feature_columns), output_path)
    return result


def run_feature_extraction(
    config: PipelineConfig,
    requested_datasets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    selected_artifacts = select_dataset_artifacts(requested_datasets)
    shared_extractor: Optional[ViTLargeFeatureExtractor] = None
    if not dry_run:
        logger = setup_logging("feature_extractor.log")
        shared_extractor = ViTLargeFeatureExtractor(
            model_name=config.model_name,
            cache_dir=config.model_cache_dir,
            device=config.device,
        )
        logger.info("Loaded model %s on device %s", config.model_name, shared_extractor.device)

    for artifacts in selected_artifacts:
        results[artifacts.name] = extract_features_for_dataset(
            metadata_path=artifacts.bronze_path,
            output_path=artifacts.silver_path,
            config=config,
            dry_run=dry_run,
            extractor=shared_extractor,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ViT-Large image embeddings into silver Excel files.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of datasets to process. Supported values: NIH TCIA RSNA.",
    )
    parser.add_argument(
        "--image-root",
        help="Optional image root directory. Use this when the metadata paths are not reachable from the project root.",
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
        help="Batch size used during feature extraction.",
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
        "--dry-run",
        action="store_true",
        help="Validate metadata and image path resolution without loading the model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        image_root=Path(args.image_root).resolve() if args.image_root else None,
        model_name=args.model_name,
        batch_size=args.batch_size,
        model_cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
        max_rows=args.max_rows,
        device=args.device,
    )
    run_feature_extraction(config=config, requested_datasets=args.datasets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
