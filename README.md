# Medical Image Classification: Pulmonary Disease Detection
### **Independent Research Project | Python & PyTorch**

This project focuses on developing a robust **Computer Vision** system to classify pulmonary diseases from chest X-ray images. By utilising **Deep Learning** architectures, the project delivers an interpretable, highly sensitive triage model that assists in medical diagnostics through automated image analysis and Explainable AI (XAI).

---

## 📊 Research Results
The project followed an iterative research path, utilizing weighted Cross-Entropy loss to balance the critical trade-off between patient safety (recall) and system alert fatigue (specificity):

| Model Configuration | Loss Weights (Normal vs Pneumonia) | Overall Accuracy | Pneumonia Recall (Sensitivity) | Missed Cases (False Negatives) | Normal Specificity (False Alarms) |
| :--- | :--- | :--- | :--- | :--- |
| **v1 (Baseline Fine-Tuned)** | Unweighted | 84.0% | 99.7% | 1 | 56.8% (FP = 101) |
| **v2 (Max Recall)** | `[1.0, 3.0]` | 84.0% | 100.0% | 0 | 58.5% (FP = 97) |
| **v3 (Balanced Attempt)** | `[1.0, 2.0]` | 86.0% | 99.7% |  | 62.0% (FP = 89) |
| **v4 (Production Optimal)** | `[1.75, 2.00]` | **87.0%** | **99.2%** | **3** | **65.4% (FP = 81)** |

**Key Breakthrough:** Transitioning from an unweighted baseline to precisely tuned class penalties allowed for direct control over the clinical decision boundary. While `v2` achieved flawless 100% recall (eliminating false negatives entirely), it triggered excessive alert fatigue. Narrowing the penalty ratio in `v4` to `[1.75, 2.00]` optimized this trade-off, securing a nearly 20% reduction in false alarms and the highest overall accuracy (87%), while safely capping missed diagnoses at a clinically acceptable 0.8% (3 out of 390 cases).

---

## 🔬 Project Overview
* **Objective:** Accurate, safety-optimized classification of pulmonary conditions (e.g., Pneumonia) using convolutional neural networks (CNNs), prioritizing recall to eliminate false negatives.
* **Architecture:** Customised **ResNet50** backbone evaluated through a standalone, lightweight local web UI built natively with **PHP, HTML, and CSS** to eliminate platform bloat.
* **Interpretability:** Integrates custom **Grad-CAM (Gradient-weighted Class Activation Mapping)** via OpenCV to generate visual heatmaps, highlighting the exact anatomical regions driving the model's diagnosis.

---

## 🛠️ Architecture & Deployment Stack
* **Model:** Pre-trained ResNet50 (ImageNet weights) with a custom 2-class linear output layer (`nn.Linear(2048, 2)`).
* **Inference Engine:** Python, PyTorch, and OpenCV for image processing, tensor operations, and bounding-box heatmap generation.
* **Interface Stack:** Standalone local web UI built natively with PHP (`php -S`), HTML, and CSS.
* **Integration:** Subprocess execution via PHP `shell_exec()`, interfacing with a modular Python execution script (`inference_script_gui.py`).

---

## 📈 Training Strategy & Production Performance
The model was fine-tuned to balance high precision with critical patient safety, utilising a weighted Cross-Entropy Loss to heavily penalise missed diagnoses.

* **Test Dataset Size:** 624 unaugmented chest radiographs (234 Normal, 390 Pneumonia).
* **Loss Function Weights:** [1.75, 2.00] (Normal vs. Pneumonia)
* **Classification Accuracy:** 87.0%
* **Pneumonia Sensitivity (Recall):** 99.2% (FN = 3)
* **Normal Specificity:** 65.3% (TN = 153, FP = 81)

### Safety & Interpretability Logic
* **≥ 75.0% Confidence:** Direct classification output (`NORMAL` or `PNEUMONIA`).
* **50.0% - 74.9% Confidence:** Flags `REQUIRES ATTENTION (Borderline)` to trigger human radiologist review.
* **Visual Triage Indicator:** Dynamic CSS color-coding (Green: Normal, Red: Pneumonia, Yellow: Borderline) scaling with the Grad-CAM intensity.

---

## 📂 Repository Structure & Version History

```text
medical-image-classification/ 
├── .git/                                   # Git Version Control History
├── data/                                   # Dataset (Source: Kaggle)
│   ├── train/                              # Training set: NORMAL / PNEUMONIA
│   ├── test/                               # Test set: NORMAL / PNEUMONIA
│   └── val/                                # Validation set: NORMAL / PNEUMONIA 
├── interface/                              # Local Web UI (HTML, CSS, PHP)
├── med_ai_env/                             # Isolated Virtual Environment
├── models/                                 # Trained Model Weights (.pth)
├── notebooks/                              # Primary Experimentation Workspaces
├── scripts/                                # Modular Script Storage 
├── src/                                    # Source Code for modular scripts
├── .gitignore                              # Instructions for files Git should ignore 
├── environment.yml                         # Reproducible Conda environment export
└── README.md                               # Project overview and documentation
```

### Model Iterations
| Iteration | Files | Objective & Adjustments |
| :--- | :--- | :--- |
| **v1** | `_v1.pth`, `_script_v1.py`, `_v1.ipynb` | Initial improved baseline script. |
| **v2** | `_v2.pth`, `_script_v2.py`, `_v2.ipynb` | Penalised False Negatives heavily `[1.0, 3.0]` to maximize recall. |
| **v3** | `_v3.pth`, `_script_v3.py`, `_v3.ipynb` | Reduced penalty `[1.0, 2.0]` to lower False Positives (alert fatigue). |
| **v4 (Prod)** | `_v4.pth`, `_script_v4.py`, `_v4.ipynb` | Optimal clinical boundary `[1.75, 2.00]`. Highest accuracy at 87.0%. |
| **GUI** | `inference_script_gui.py` | Uses production `v4` weights; formatted for local web UI integration. |

---

## 🚀 Local Deployment Guide
To run the diagnostic interface locally without a heavy framework:

1. Ensure the `med_ai_env` conda environment is active in your terminal.
2. Navigate to the interface directory: `cd medical-image-classification/interface`
3. Boot the local PHP development server: `php -S localhost:8000`
4. Access the diagnostic tool via `http://localhost:8000` in your web browser.

---

## 📊 Methodology & Research Goals
Detailed technical documentation tracks all experimental iterations to prepare findings for academic review at **Queen Mary University of London**. The project follows professional **Data Governance** standards mirrored from industry experience at **NovoPart Ltd.** to ensure data integrity, IP security, and scalable system architecture.

---

## 🛡️ Research Ethics & Academic Integrity
* This project is an independent research endeavour.
* All methodologies follow standard research ethics regarding medical data handling and algorithmic transparency.

---

## 🌐 Connect with Me
* **Email:** [monica.duarte@monicaduarte.com](mailto:monica.duarte@monicaduarte.com)
* **Portfolio:** [monicaduarte.com](https://monicaduarte.com)
* **LinkedIn:** [linkedin.com/in/monicaduarteai](https://linkedin.com/in/monicaduarteai)
* **GitHub:** [github.com/monicaduarteai](https://github.com/monicaduarteai)

---
*This project is a core component of my development in Computer Vision and Artificial Intelligence.*
