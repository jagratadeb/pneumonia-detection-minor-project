# Pneumonia Detection from Chest X-ray Images

**A Comparative Study of Machine Learning and Deep Learning Approaches for Binary Classification**

---

## Executive Summary

This project presents a comprehensive comparative analysis of machine learning and deep learning models for automated pneumonia detection from chest X-ray images. The study implements both traditional machine learning approaches (KNN, SVM) and state-of-the-art convolutional neural networks (ResNet-18, ResNet-34, ResNet-50, VGGNet-16) using the COVID-19 Radiography Database. The objective is to evaluate and compare model performance across standardized evaluation metrics to identify the most effective approach for pneumonia classification in clinical settings.

**Repository:** https://github.com/jagratadeb/pneumonia-detection-minor-project  
**Dataset:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

---

## 1. Introduction

### 1.1 Problem Statement

Pneumonia detection from chest radiographs is a critical task in medical imaging that can benefit significantly from automated analysis. This study performs **binary classification** of chest X-ray images to distinguish between:

- **Class 0:** Normal (healthy)
- **Class 1:** Pneumonia (Viral Pneumonia, Lung Opacity, COVID-19)

The goal is to develop reliable automated detection systems that can assist radiologists in diagnosis and patient screening.

### 1.2 Methodology Overview

The study employs a systematic approach comparing two distinct paradigms:
1. **Machine Learning Models:** Traditional algorithms with hand-crafted feature extraction
2. **Deep Learning Models:** End-to-end trainable convolutional neural networks

All models are trained and evaluated using identical dataset splits and standardized evaluation metrics to ensure fair comparison.

---

## 2. Dataset

### 2.1 Dataset Description

**Source:** COVID-19 Radiography Database (Kaggle)

The dataset comprises chest X-ray images categorized into two classes: normal and pneumonia-affected. All images are preprocessed and standardized for computational consistency.

### 2.2 Data Stratification

The dataset is divided into three mutually exclusive subsets:

| Subset | Percentage | Purpose |
|--------|-----------|---------|
| Training | 70% | Model training and parameter optimization |
| Validation | 15% | Hyperparameter tuning and early stopping |
| Testing | 15% | Final performance evaluation |

### 2.3 Data Preprocessing

All images undergo standardized preprocessing:
- **Resizing:** 224 × 224 pixels (uniform input dimension)
- **Normalization:** Pixel value normalization to [0, 1] range
- **Feature Scaling:** StandardScaler applied for machine learning models

---

## 3. Methodology

### 3.1 Machine Learning Models

Machine learning models utilize hand-crafted features extracted from chest X-ray images. Model-agnostic feature extraction is applied to all ML approaches.

#### 3.1.1 Feature Extraction

Two feature extraction methods are supported:
- **HOG (Histogram of Oriented Gradients):** Captures edge and texture information
- **CNN Features:** Pretrained ResNet-18 backbone features for learned representations

**Mandatory Preprocessing:**
- Resize images to 224 × 224 pixels
- Apply StandardScaler normalization

#### 3.1.2 K-Nearest Neighbors (KNN)

| Parameter | Value | Notes |
|-----------|-------|-------|
| k values | 3, 5, 7 | Validation across multiple values |
| Distance metric | Euclidean | L2 norm distance |
| Weights | Uniform | Equal weight for all neighbors |
| Scaling | StandardScaler | Mandatory normalization |

**Evaluation Metrics:** Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC, Confusion Matrix

#### 3.1.3 Support Vector Machine (SVM)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | RBF (Radial Basis Function) | Non-linear decision boundary |
| Regularization (C) | 1.0 | Default regularization strength |
| Gamma | scale | Automatic kernel coefficient |
| Scaling | StandardScaler | Mandatory normalization |

**Evaluation Metrics:** Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC, Confusion Matrix

---

### 3.2 Deep Learning Models

Deep learning models employ convolutional neural networks (CNNs) based on pretrained architectures from ImageNet. All models are fine-tuned using the same training protocol for fair comparison.

#### 3.2.1 Common Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Adaptive learning rate |
| Batch Size | 32 | Images per training iteration |
| Epochs | 20 | Maximum training iterations |
| Early Stopping | Yes | Patience = 5 epochs |
| Loss Function | Cross-Entropy | Standard classification loss |

#### 3.2.2 ResNet-18

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Pretraining | ImageNet |
| Learning Rate | 0.001 |
| Final Layer | 2-class classifier |
| Input Shape | 3 × 224 × 224 |

#### 3.2.3 ResNet-34

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-34 |
| Pretraining | ImageNet |
| Learning Rate | 0.001 |
| Final Layer | 2-class classifier |
| Input Shape | 3 × 224 × 224 |

**Note:** Configuration identical to ResNet-18 for direct architectural comparison.

#### 3.2.4 ResNet-50

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | ResNet-50 | Deeper network with bottleneck blocks |
| Pretraining | ImageNet | Transfer learning |
| Learning Rate | 0.0005 | Lower learning rate for stability |
| Final Layer | 2-class classifier | Binary classification output |
| Input Shape | 3 × 224 × 224 | Standard input dimension |

#### 3.2.5 VGGNet-16

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | VGG-16 | Deep sequential architecture |
| Pretraining | ImageNet | Transfer learning |
| Learning Rate | 0.0005 | Conservative learning rate |
| Dropout | 0.5 | Regularization in classifier |
| Final Layer | 2-class classifier | Binary classification output |
| Batch Size | 32 | Standard batch |
| Epochs | 20 | Standard training duration |

---

## 4. Performance Metrics

### 4.1 Evaluation Metrics Definition

All models are evaluated using the following standardized metrics:

- **Accuracy:** Proportion of correct predictions among total predictions
- **Precision:** Proportion of true positives among predicted positives
- **Recall (Sensitivity):** Proportion of true positives among actual positives
- **Specificity:** Proportion of true negatives among actual negatives
- **F1-score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve

### 4.2 Model Performance Summary

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|-------------|----------|---------|
| KNN (k=7) | 0.9717 | 0.8696 | 0.8911 | 0.9823 | 0.8802 | 0.9904 |
| SVM (RBF) | — | — | — | — | — | — |
| ResNet-18 | 0.9896 | 0.9466 | 0.9653 | 0.9928 | 0.9559 | 0.9992 |
| ResNet-34 | 0.9902 | 0.9512 | 0.9653 | 0.9935 | 0.9582 | 0.9992 |
| ResNet-50 | — | — | — | — | — | — |
| VGGNet-16 | — | — | — | — | — | — |

---

## 5. Results and Analysis

### 5.1 Key Findings

**Completed Model Evaluations:**

- **KNN (k=7):** Achieved 97.17% accuracy with excellent ROC-AUC of 0.9904. The model demonstrates strong performance as a baseline traditional ML approach.

- **ResNet-18:** Attained 98.96% accuracy with superior precision (0.9466) and ROC-AUC (0.9992), indicating minimal false positives.

- **ResNet-34:** Marginally outperformed ResNet-18 with 99.02% accuracy and 0.9582 F1-score, demonstrating the benefit of increased network depth.

**Observations:**

- Deep learning models consistently outperform traditional machine learning approaches
- Increased ResNet depth (18 → 34) provides incremental performance improvements
- All completed models achieve >97% accuracy, indicating strong classification capability

### 5.2 Visualization Artifacts

**Generated for all models:**
- Confusion Matrix Heatmaps
- ROC Curves

**Generated for deep learning models:**
- Training Loss vs. Epochs
- Validation Loss vs. Epochs
- Training Accuracy vs. Epochs
- Validation Accuracy vs. Epochs
- Grad-CAM Heatmaps (Optional)

---

## 6. Standardization Protocol

To ensure methodological rigor and fair model comparison, the following standardization rules are enforced:

1. **Consistent Dataset Split:** All models utilize identical 70-15-15 train-validation-test split
2. **Test Set Integrity:** Test set remains unused until final evaluation phase
3. **Unified Preprocessing:** Identical preprocessing pipeline across all models
4. **Standard Evaluation:** All models evaluated on six core metrics (Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC)
5. **Reproducibility:** Standardized hyperparameters and training protocols documented

---

## 7. Conclusion

This study presents a systematic comparison of machine learning and deep learning approaches for pneumonia detection from chest X-ray images. Preliminary results from completed evaluations demonstrate that:

1. Deep learning models (ResNet variants) substantially outperform traditional ML approaches in classification accuracy and robustness
2. ResNet-34 emerges as the current best-performing model with 99.02% accuracy
3. All evaluated models show promise for clinical application

Further evaluation of remaining models (SVM, ResNet-50, VGGNet-16) will provide additional insights into the trade-offs between model complexity, computational efficiency, and classification performance.

---

## Contributors

- Jagrata Deb
- Baibhab Majumder
- Anik Roy
- Ayushman Nanda
- Ayush Pradhan

---

## References

1. Tawsifur Rahman, et al. "COVID-19 Radiography Database." Kaggle, 2020.
   https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

> To be updated soon.
