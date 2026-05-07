from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import sys
from pathlib import Path
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure project root is on sys.path so `models.scripts` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.scripts import ft_inference
except Exception as e:  # pragma: no cover - import may fail if artifacts are missing
    ft_inference = None
    ft_inference_import_error = e
    print("[WARN] Unable to import FT inference helpers:", e)


@app.get("/model_info")
def model_info():
    try:
        if ft_inference is None:
            raise RuntimeError(str(ft_inference_import_error))
        info = ft_inference.model_info_snapshot()
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if ft_inference is None:
            raise_runtime = RuntimeError(str(ft_inference_import_error))
            raise raise_runtime

        image = Image.open(file.file).convert("RGB")

        # Use the project's FT inference helper which ties together ViT feature extraction
        # and the FT-Transformer classifier (loads artifacts from artifacts/models)
        results, feature_snapshot = ft_inference.predict_from_pil_image(image, top_k=3, image_name=file.filename)

        # results is a list (one entry per image). We sent a single image.
        response = results[0] if results else {}

        # Convert feature snapshot (pandas.DataFrame) to list of dicts if available
        try:
            features = feature_snapshot.to_dict(orient="records")
        except Exception:
            features = None

        payload = {
            "prediction": response.get("predicted_label"),
            "confidence": response.get("confidence"),
            "confidence_band": response.get("confidence_band"),
            "confidence_note": response.get("confidence_note"),
            "top_predictions": response.get("top_predictions"),
            "gradcam": response.get("gradcam"),
            "gradcam_method": response.get("gradcam_method"),
            "model_pipeline": {
                "feature_extraction": ft_inference.load_runtime_bundle().feature_extractor_model_name,
                "classification": "FT-Transformer",
            },
            "feature_snapshot": features,
        }
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)
