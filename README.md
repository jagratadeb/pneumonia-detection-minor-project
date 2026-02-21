# 🫁 Pneumonia Detection – Minor Project

**GitHub Repository:**  
https://github.com/jagratadeb/pneumonia-detection-minor-project

---

## 1️⃣ Problem Definition

This project performs **Binary Classification** on chest X-ray images.

### Classes:
- **Class 0 → Normal**
- **Class 1 → Pneumonia** (Viral Pneumonia, Lung Opacity, COVID)
 
---

## 2️⃣ Dataset

**Dataset Used:** COVID-19 Radiography Database  

### Standardized Split:

| Split | Percentage |
|-------|------------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Rules:
- Test set must remain untouched until final evaluation  
- Same dataset split must be used for ALL models  

---

# 🧠 Machine Learning Models

⚠ ML models require feature extraction first.

---

## 🔹 Feature Extraction (Common for ML)

Use one of the following:

- HOG (Histogram of Oriented Gradients)  
- OR Pretrained CNN Features (ResNet18 backbone)

### Standardization Steps:
- Resize images → 224 × 224
- Normalize pixel values
- Apply StandardScaler (Mandatory)

---

## 🔹 ML Model 1: KNN (K-Nearest Neighbors)

### Hyperparameters:
- k = 5 (validate with 3, 5, 7)
- Distance = Euclidean
- Weights = Uniform
- StandardScaler = Mandatory

### Testing Parameters:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🔹 ML Model 2: SVM (Support Vector Machine)

### Hyperparameters:
- Kernel = RBF
- C = 1
- Gamma = scale
- StandardScaler = Mandatory

### Testing Parameters:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

# 🤖 Deep Learning Models

Use pretrained architectures for fair comparison.

---

## 🔹 Model 1: ResNet-18

- Pretrained = ImageNet
- Replace final FC layer
- Output = 2 classes

### Training Parameters:
- Optimizer = Adam
- Learning Rate = 0.001
- Batch Size = 32
- Epochs = 20
- Early Stopping (patience = 5)

### Testing Parameters:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training / Validation Curves

---

## 🔹 Model 2: ResNet-34

Same parameters as ResNet-18.

---

## 🔹 Model 3: ResNet-50

Same as above but:
- Learning Rate = 0.0005 (deeper network)

---

## 🔹 Model 4: VGGNet-16

- Pretrained = ImageNet
- Replace last classifier layer
- Dropout = 0.5

### Training:
- Learning Rate = 0.0005
- Batch Size = 32
- Epochs = 20

---

# 📊 Evaluation Metrics (All Models)

For every model compute:

1. Accuracy  
2. Precision  
3. Recall (Sensitivity)  
4. Specificity  
5. F1-score  
6. ROC-AUC  

---

# 📈 Required Graphs

## For ALL Models:
- Confusion Matrix Heatmap
- ROC Curve

## For DL Models Only:
- Training Loss vs Epoch
- Validation Loss vs Epoch
- Training Accuracy vs Epoch
- Validation Accuracy vs Epoch
- Grad-CAM Heatmap (Optional but recommended)

---

# ✅ Standardization Rules Summary

- Same dataset split across all models
- Test set evaluated only once (final stage)
- Same preprocessing pipeline for fairness
- Report all required metrics and graphs
