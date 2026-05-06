from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
import torchxrayvision as xrv
import cv2
import base64
import io
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Model
# =========================
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"[OK] Model loaded successfully")
print(f"[DEVICE] Device: {device}")

# Test with random noise to see if model produces varying outputs
test_input = torch.randn(1, 1, 224, 224).to(device)
with torch.no_grad():
    test_output = model(test_input)[0]
print(f"[TEST] Random input test output: {test_output.cpu().numpy()}")

# =========================
# Class Mapping (8 classes)
# =========================
mapping = {
    "Pneumonia": "Pneumonia",
    "Effusion": "Effusion",
    "Mass": "Mass/Nodule",
    "Nodule": "Mass/Nodule",
    "Cardiomegaly": "Cardiomegaly",
    "Pneumothorax": "Pneumothorax",
    "Atelectasis": "Atelectasis",
    "No Finding": "Normal"
}

pathologies = model.pathologies

# =========================
# Preprocess
# =========================
def preprocess(img: Image.Image):
    img = img.convert("L")  # grayscale (1 channel)
    img = img.resize((224, 224))

    img = np.array(img).astype(np.float32)
    
    # Use torchxrayvision's normalization
    img = xrv.datasets.normalize(img, maxval=255)
    
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    tensor = torch.from_numpy(img).to(device)
    return tensor

# =========================
# Grad-CAM
# =========================
target_layers = [model.features[-1]]

def generate_gradcam(input_tensor, class_idx):
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return grayscale_cam

# =========================
# Overlay Heatmap
# =========================
def overlay_heatmap(img: Image.Image, heatmap):
    img = img.convert("RGB")
    img = np.array(img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# =========================
# Endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"[REQUEST] Predict request received: filename={file.filename} content_type={file.content_type}")
    try:
        image = Image.open(file.file)
        print(f"Loaded image: format={image.format} size={image.size} mode={image.mode}")

        input_tensor = preprocess(image)
        print(f"Input tensor shape: {list(input_tensor.shape)}")

        with torch.no_grad():
            outputs = model(input_tensor)[0]

        outputs_np = outputs.cpu().numpy()
        print(f"Model outputs: {outputs_np.tolist()}")

        # =========================
        # Map to 8 classes
        # =========================
        scores = {}

        for i, pathology in enumerate(pathologies):
            if pathology in mapping:
                mapped = mapping[pathology]
                scores[mapped] = max(scores.get(mapped, 0), outputs_np[i])

        print(f"Mapped scores: {scores}")

        # =========================
        # Get best class
        # =========================
        prediction = max(scores, key=scores.get)
        confidence = float(scores[prediction])
        print(f"Prediction: {prediction} | Confidence: {confidence*100:.2f}%")

        # =========================
        # Grad-CAM index
        # =========================
        class_idx = pathologies.index(
            next(p for p in pathologies if mapping.get(p) == prediction)
        )

        heatmap = generate_gradcam(input_tensor, class_idx)
        overlay = overlay_heatmap(image, heatmap)

        # =========================
        # Convert to base64
        # =========================
        _, buffer = cv2.imencode(".png", overlay)
        img_str = base64.b64encode(buffer).decode()

        return JSONResponse({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "gradcam": f"data:image/png;base64,{img_str}"
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)