# Project Report: Wrist & Hand Fracture Detection using YOLOv8

**Course:** Computer Vision  
**Submission Type:** Bring Your Own Project (BYOP)  
**Author:** Aditya  
**Date:** March 2026

---

## 1. Problem Statement

Bone fractures of the wrist and hand are among the most common injuries seen in emergency departments worldwide. Accurate and timely detection is critical — a missed fracture can lead to improper immobilisation, prolonged pain, and long-term functional loss. Yet diagnosis depends entirely on a radiologist interpreting X-ray images, a process that is time-consuming, resource-intensive, and subject to human fatigue.

This project addresses that gap by building an automated fracture detection system using object detection, capable of localising fractures in wrist and hand X-ray images and presenting findings with bounding box annotations and confidence scores.

---

## 2. Personal Motivation

This project was personally motivated. Following a road accident, I underwent X-ray imaging for a suspected wrist fracture. The experience brought to my attention how critically the diagnosis process depends on radiologist availability and attention — factors that vary significantly across institutions and geographies. In smaller hospitals and urgent care facilities, specialist radiologists may not always be available immediately.

This personal encounter shaped the direction of this project: rather than solving an abstract problem, I chose to build something with genuine practical relevance, applying computer vision techniques from this course to a domain I now understand firsthand.

---

## 3. Proposed Solution

The solution is a supervised object detection pipeline that:

1. Takes an X-ray image of the wrist/hand as input
2. Identifies regions containing fractures
3. Outputs bounding box coordinates around each detected fracture with an associated confidence score

The system is intended as a **decision-support tool** — not a replacement for clinical judgement, but a screening aid that could flag suspicious regions for a radiologist's review, particularly useful in high-throughput or under-resourced settings.

---

## 4. Technical Approach

### 4.1 Model Selection — YOLOv8

I selected **YOLOv8** (You Only Look Once, version 8) by Ultralytics as the detection backbone. The rationale for this choice:

- **Speed:** YOLOv8 processes images in a single forward pass, making it suitable for real-time screening
- **Accuracy:** YOLOv8s achieves competitive mAP on standard benchmarks while remaining lightweight enough to train on consumer GPUs
- **Pre-training:** Starting from COCO-pretrained weights allows the model to leverage learned low-level features (edges, textures) that transfer well to X-ray imagery
- **Ecosystem:** The Ultralytics library provides clean training, evaluation, and export APIs that integrate naturally with Jupyter notebooks

I chose the **YOLOv8s (small)** variant — a deliberate trade-off between the computational constraints of an academic project and the accuracy demands of medical imaging.

### 4.2 Dataset

The dataset was sourced from Kaggle and consists of annotated wrist and hand X-ray images in YOLO format, with bounding box labels marking fracture regions. The dataset is split into training, validation, and test sets.

Key dataset properties explored during EDA:
- Class distribution (single class: fracture)
- Bounding box size distribution (used to verify annotations are appropriately sized relative to images)
- Sample visualisation to confirm annotation quality

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s |
| Epochs | 50 (early stopping at patience=15) |
| Image size | 640×640 |
| Batch size | 16 |
| Optimiser | AdamW |
| Learning rate | 1e-3 (cosine decay to 1e-5) |
| Warmup epochs | 3 |
| Augmentation | Mosaic, horizontal flip, rotation (±10°), HSV jitter |

The rotation augmentation was a deliberate addition: X-rays can be taken at varying angles depending on patient positioning, so the model needed to be robust to orientation variation.

### 4.4 Evaluation Metrics

The model was evaluated using standard object detection metrics:

- **mAP@0.5** — mean Average Precision at IoU threshold 0.5 (primary metric)
- **mAP@0.5:0.95** — stricter metric averaged across multiple IoU thresholds
- **Precision** — fraction of detections that are correct
- **Recall** — fraction of actual fractures that were detected

In a clinical context, **recall is prioritised over precision** — it is more harmful to miss a fracture (false negative) than to flag a normal bone (false positive) that a radiologist then reviews and dismisses.

---

## 5. Key Design Decisions

### Why object detection over classification?
A classification model would only answer "is there a fracture?" — it would not tell us *where* the fracture is. Object detection provides spatial localisation, which is far more useful for a clinician who needs to know the specific site of the injury. This also makes the model's output interpretable and trustworthy.

### Why YOLOv8 over Faster R-CNN?
Faster R-CNN achieves higher accuracy on small objects but is significantly slower (two-stage detector). For a screening tool where throughput matters, YOLOv8's single-stage detection is a better fit. Additionally, YOLOv8's Ultralytics API is better documented and easier to iterate with, which was important given the time constraints of an academic project.

### Pre-trained vs. from scratch?
Transfer learning from COCO pre-trained weights was used. Training from scratch on a relatively small medical imaging dataset would have led to significant overfitting. The COCO weights provide a strong initialisation of low-level feature detectors (gradients, edge detectors) that are genuinely useful for X-ray feature extraction.

### Data augmentation choices
Standard YOLO augmentations (mosaic, colour jitter) were kept enabled. Rotation was explicitly included since X-ray patient positioning introduces orientation variability. Horizontal flips were also enabled since left/right hand symmetry makes flipped images plausible training examples.

---

## 6. Challenges Faced

**Dataset quality and annotation noise:** Medical imaging datasets from public sources can contain inconsistent annotation quality. Some bounding boxes were found to be either too large (encompassing non-fracture regions) or too small. This required careful EDA before training rather than blindly fitting the model.

**Class imbalance between images:** Not all X-ray images in the dataset contain fractures. Handling the ratio of positive to negative images required attention during training to ensure the model did not become biased toward predicting "no fracture."

**Confidence threshold calibration:** The default confidence threshold of 0.25 produced false positives on normal bone structures. A threshold of 0.35–0.45 was found to better balance precision and recall on the validation set. This is an important deployment consideration — the threshold should be tuned based on clinical context.

**Transfer from natural images to X-rays:** COCO pre-training is on natural RGB images. X-rays are single-channel grayscale images with inverted intensity characteristics. YOLOv8 handles this reasonably by accepting grayscale inputs, but the domain gap does affect initial training convergence.

---


## 7. Limitations & Future Work

**Current limitations:**
- The model is trained only on wrist/hand X-rays and cannot generalise to other bone types
- Performance on X-rays from different imaging equipment or protocols has not been tested
- The model has not been validated against clinical ground truth from radiologists
- No explainability mechanism (e.g., Grad-CAM) is integrated into the current pipeline

**Future improvements:**
- Extend to multi-class detection (fracture types: hairline, displaced, comminuted)
- Integrate Grad-CAM visualisations to highlight features the model attends to
- Build a lightweight web interface (Flask/Streamlit) for interactive inference
- Validate against radiologist annotations for clinical credibility

---

## 8. Learning Outcomes

This project deepened my understanding in several areas:

- **Object detection architecture:** Working directly with YOLOv8 clarified the difference between single-stage and two-stage detectors, and the role of anchor-free detection in modern YOLO variants
- **Transfer learning:** Understanding *why* and *how* pre-trained weights accelerate convergence in a domain-shifted setting
- **Evaluation literacy:** Learning to interpret mAP curves, confusion matrices, and PR curves rather than relying on a single accuracy number
- **Medical imaging specifics:** Developing sensitivity to the unique properties of X-ray data — contrast, noise, orientation — and how augmentation choices must reflect clinical reality
- **End-to-end ML workflow:** Planning and executing an entire pipeline from raw data to trained, exportable model

---

## 9. References

1. Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics  
2. Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767  
3. Rajpurkar, P. et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays*. arXiv:1711.05225 *(referenced for transfer learning on medical imaging)*  
4. Dataset: [Kaggle — Bone Fracture Detection] *(insert full dataset citation here)*  
5. Buslaev, A. et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information, 11(2), 125.

---

*This report was prepared as part of the Computer Vision BYOP submission. All code is available in the linked GitHub repository.*
