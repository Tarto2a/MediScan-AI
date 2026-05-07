from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import ARTIFACT_REPORTS_ROOT, setup_logging  # noqa: E402
from scripts.ft_inference import load_input_frame, missing_artifacts, predict_from_frame  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with the saved FT-Transformer model on a feature table or single sample.")
    parser.add_argument("--input-path", help="Path to a CSV/XLSX feature table.")
    parser.add_argument("--sample-json", help="Path to a JSON file containing a single sample dictionary.")
    parser.add_argument("--output-path", default=str(ARTIFACT_REPORTS_ROOT / "ft_test_predictions.csv"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--row-index", type=int, help="Optional row index to score from --input-path.")
    return parser


def load_json_sample(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError("--sample-json must contain a single JSON object.")


def resolve_input_frame(input_path: Optional[str], sample_json: Optional[str], row_index: Optional[int]) -> pd.DataFrame:
    if input_path:
        return load_input_frame(Path(input_path), row_index=row_index)
    if sample_json:
        return load_json_sample(Path(sample_json))
    raise ValueError("Provide either --input-path or --sample-json.")


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("test_ft_transformer.log")
    missing = missing_artifacts()
    if missing:
        raise FileNotFoundError("Required FT-Transformer artifacts are missing:\n" + "\n".join(missing))

    input_frame = resolve_input_frame(args.input_path, args.sample_json, args.row_index)
    predictions = predict_from_frame(input_frame=input_frame, top_k=args.top_k)

    rows = []
    for row_index, prediction in enumerate(predictions):
        row_payload = dict(prediction["metadata"])
        row_payload.update(
            {
                "row_index": row_index,
                "predicted_label": prediction["predicted_label"],
                "confidence": float(prediction["confidence"]),
                "confidence_band": prediction["confidence_band"],
                "top_predictions": json.dumps(prediction["top_predictions"]),
            }
        )
        rows.append(row_payload)

    output_frame = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)

    logger.info("Generated %s FT-Transformer predictions.", len(output_frame))
    print(output_frame.head().to_string(index=False))
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
