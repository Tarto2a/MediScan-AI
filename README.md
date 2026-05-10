# Chest Image Analysis Web App

This project is a local clinical AI demo for chest image classification. It lets a user upload a chest X-ray or CT slice, sends the image to a FastAPI backend, extracts ViT image features, and classifies the result with a saved FT-Transformer model.

The app returns the predicted class, confidence, top ranked classes, and a ViT attention overlay when available.

## Tech Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI
- ML runtime: PyTorch, Transformers, scikit-learn, pandas, Pillow
- Model pipeline: ViT-Large feature extraction + FT-Transformer classifier

## Main Project Structure

```text
backend/
  server.py                 FastAPI API used by the frontend

frontend/
  src/                      React UI

models/
  scripts/                  Inference and model helper code
  artifacts/models/         Required saved model files
  training/                 Large training data, reports, and old artifacts
```

## Required Runtime Model Files

Keep these files in `models/artifacts/models/`:

- `best_ft_transformer.pth`
- `ft_preprocessor.joblib`
- `label_mapping.json`
- `selected_features.json`
- `preprocessing_config.json`
- `inference_config.json`

Without these files, `/predict` cannot run.

## Run The App

Install frontend dependencies if needed:

```bash
npm install
```

Start frontend and backend together:

```bash
npm start
```

The root `package.json` runs:

- frontend: `cd frontend && npm run dev`
- backend: `cd backend && uvicorn server:app --reload`

## API Endpoints

- `GET /model_info` returns model and pipeline metadata
- `POST /predict` accepts an uploaded image file and returns classification results

## Notes

The `models/training/` folder contains large datasets, generated reports, plots, and training artifacts. It is intentionally ignored by Git.

This project is for research/demo use and should not be used as a real medical diagnosis system without proper clinical validation.
