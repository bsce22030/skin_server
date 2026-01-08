from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

# -------------------------
# App setup
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Flutter
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

# -------------------------
# Labels
# -------------------------
idx_to_label = {
    0: "Actinic Keratosis (ACK)",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Melanoma (MEL)",
    3: "Nevus (NEV)",
    4: "Squamous Cell Carcinoma (SCC)",
    5: "Seborrheic Keratosis (SEK)",
}

# -------------------------
# Load model
# -------------------------
ckpt = torch.load("best_model.pt", map_location=device)

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    ckpt["num_classes"]
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# Image transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]

    results = {
        idx_to_label[i]: round(float(p * 100), 2)
        for i, p in enumerate(probs)
    }

    pred_idx = probs.argmax().item()

    return {
        "prediction": idx_to_label[pred_idx],
        "confidence": results[idx_to_label[pred_idx]],
        "all_probabilities": results,
    }
