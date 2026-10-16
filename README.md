# 🚨 RescueVision — Disaster Damage & Rescue Priority AI Wrapper

Production-ready inference wrapper for the **RescueVision** ResNet50 Transfer Learning model. It classifies post-disaster satellite imagery into `damage` vs `no_damage` and assigns rescue operational priority levels (`P1 - CRITICAL` to `P4 - LOW`).

---

## 📥 1. Getting Your `.keras` File from Google Colab

Once training completes in your Colab notebook, run this cell to download the model file to your computer:

```python
from google.colab import files
files.download('rescuevision_model.keras')
```

Place `rescuevision_model.keras` directly in your project root directory:
```
rescuevision/
├── rescuevision_model.keras   <-- Place your downloaded file here!
├── model_wrapper.py
├── example_usage.py
└── rescuevision/
```

---

## 🚀 2. Quickstart Code

```python
from model_wrapper import RescueVisionModel

# 1. Initialize model (automatically finds 'rescuevision_model.keras')
model = RescueVisionModel()

# 2. Predict on an image (filepath, PIL image, bytes, or numpy array)
result = model.predict("path/to/satellite_crop.jpeg")

# 3. View assessment
print(result.summary())
# 🔴 [DAMAGE] (94.2% damage confidence) | Priority: P1 - CRITICAL -> Immediate rescue deployment

# 4. Export to JSON (for REST APIs, databases, or dashboards)
print(result.to_json())
```

---

## ⚡ 3. Batch Inference

Process multiple crops simultaneously:

```python
images = [
    "crop_1.jpeg",
    "crop_2.jpeg",
    "crop_3.jpeg"
]

results = model.predict_batch(images, batch_size=32)
for res in results:
    print(res.summary())
```

---

## 📊 4. Generate Visual Inspection Card

Create an annotated inspection card with classification, priority badge, and operational action recommendation:

```python
model.visualize("crop_1.jpeg", save_path="assessment_card.png")
```

---

## 🌐 5. Launch the Web Application

RescueVision features a full-featured Tactical Emergency Dashboard built with **Streamlit** (frontend) and **FastAPI** (REST backend).

### 🚀 One-Command Local Launch
Run both the FastAPI backend and Streamlit dashboard simultaneously:
```bash
python run_app.py
```
- **Streamlit Web Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Features in the Web App:
1. **🛰️ Live Crop Assessment**: Upload any satellite photo or test preloaded buildings to get damage detection, priority tags, and downloadable assessment report cards.
2. **📋 Batch Reconnaissance**: Bulk upload hundreds of building tiles to view total casualty risk, sector damage rates, and export CSV triage manifests for field rescue units.
3. **🗺️ Geospatial Tactical Map**: Interactive 3D aerial map color-coded by rescue priority (🔴 P1 Critical, 🟠 P2 High, 🟡 P3 Medium, 🟢 P4 Low).
4. **📖 Model Architecture**: Full ResNet50 neural network specs, benchmarks, and training metrics.

---

## 🐳 6. Deploy Anywhere (Docker & Cloud)

### Run with Docker Compose:
```bash
docker-compose up --build
```

### Deploy to Hugging Face Spaces / Streamlit Cloud / Render:
- **Repository Type**: Streamlit
- **Main file path**: `frontend/app.py`
- **Requirements file**: `requirements.txt`

---

## 💻 7. Command Line Interface (CLI)

```bash
# Analyze an image using your trained model
python model_wrapper.py path/to/image.jpeg

# Generate visual report image
python model_wrapper.py path/to/image.jpeg --visualize

# Test without weights (Mock Mode) while Colab is training
python model_wrapper.py path/to/image.jpeg --mock
```

---

## 🏷️ 6. Rescue Priority System

| Confidence (Damage) | Priority Level | Action |
|---|---|---|
| **> 90%** | 🔴 **P1 - CRITICAL** | Immediate rescue deployment |
| **70% – 90%** | 🟠 **P2 - HIGH** | Urgent assessment needed |
| **50% – 70%** | 🟡 **P3 - MEDIUM** | Schedule inspection |
| **< 50%** | 🟢 **P4 - LOW** | No immediate action |

---

## 📍 7. Automated GPS Geolocation
Satellite crops formatted as `<lon>_<lat>.jpeg` (e.g., `-95.548123_29.756162.jpeg`) automatically extract geospatial coordinates into `result.latitude` and `result.longitude` for GIS mapping.
