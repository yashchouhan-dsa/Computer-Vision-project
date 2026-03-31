# 🦴 Wrist & Hand Fracture Detection using YOLOv8

A computer vision project that uses **YOLOv8 object detection** to automatically localise fractures in wrist and hand X-ray images, built as part of a Computer Vision course capstone (BYOP).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Fracture diagnosis from X-ray images depends on radiologist availability — a bottleneck in under-resourced or high-throughput medical settings. This project builds an automated detection pipeline that localises fractures in wrist/hand X-rays and outputs bounding box annotations with confidence scores, functioning as a screening aid.

---

## 🗂 Project Structure

```
fracture-detection/
│
├── fracture_detection.ipynb   # Main notebook: EDA → Training → Evaluation → Inference
├── data.yaml                  # Auto-generated YOLO dataset config
├── report.md                  # Full project report
├── requirements.txt           # Python dependencies
│
├── data/
│   └── raw/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── fracture_det/              # Created after training
│   └── yolov8s_wrist/
│       ├── weights/
│       │   ├── best.pt        # Best model checkpoint
│       │   └── last.pt
│       ├── results.csv
│       ├── PR_curve.png
│       └── confusion_matrix.png
│
└── outputs/                   # Saved figures from the notebook
    ├── eda_overview.png
    ├── sample_images.png
    ├── training_curves.png
    └── inference_results.png
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.9 or higher
- pip
- A CUDA-capable GPU is strongly recommended (CPU training will be very slow)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fracture-detection.git
cd fracture-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

1. Go to the Kaggle dataset page: *(insert your dataset URL here)*
2. Download and extract it into `data/raw/`
3. Ensure the folder structure matches the layout shown above (YOLO format with `images/` and `labels/` subdirectories)

> **Note:** If your dataset uses Pascal VOC XML or CSV annotations, Section 3 of the notebook includes a conversion utility.

---

## 🚀 Usage

### Run the full pipeline

Open `fracture_detection.ipynb` in Jupyter and run all cells top to bottom:

```bash
jupyter notebook fracture_detection.ipynb
```

The notebook is self-contained and walks through:

| Section | What it does |
|---------|-------------|
| 1. Setup | Installs dependencies, checks GPU |
| 2. EDA | Visualises class distribution, bbox sizes, sample images |
| 3. Preparation | Generates `data.yaml` for YOLO, previews augmentations |
| 4. Training | Trains YOLOv8s with configurable hyperparameters |
| 5. Evaluation | Computes mAP, precision, recall; plots training curves |
| 6. Inference | Runs detection on test images, displays results |
| 7. Export | Exports model to ONNX format |

### Run inference on a single image

```python
from ultralytics import YOLO

model = YOLO("fracture_det/yolov8s_wrist/weights/best.pt")
results = model.predict("path/to/xray.jpg", conf=0.35)
results[0].show()
```

---

## 📊 Results

*(Update with your actual results after training)*

| Metric | Value |
|--------|-------|
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |
| Precision | — |
| Recall | — |

---

## 🛠 Configuration

Key parameters you can change at the top of Section 4 in the notebook:

```python
MODEL_VARIANT = "yolov8s.pt"   # yolov8n (fastest) → yolov8m (most accurate)
EPOCHS        = 50
IMG_SIZE      = 640
BATCH_SIZE    = 16             # Reduce to 8 if you hit GPU memory errors
CLASS_NAMES   = ["fracture"]   # Add more classes if your dataset has them
```

---

## 📦 Requirements

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision
opencv-python-headless
matplotlib
seaborn
pandas
numpy
tqdm
Pillow
PyYAML
albumentations
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. The model has not been clinically validated and should not be used for medical diagnosis. All clinical decisions must be made by qualified healthcare professionals.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Kaggle dataset contributors
- Course instructors and teaching staff
