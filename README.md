
# 🧠 Brain Tumor Classification Using Transfer Learning (Xception)

> **Dataset**: MRI Brain Scans (4 Classes: Glioma, Meningioma, Pituitary, No Tumor)

---

## 📌 Project Description

This project presents a deep learning pipeline to classify brain tumors using MRI scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. It leverages **transfer learning** with the **Xception model** as the feature extractor, followed by fully connected layers for classification.

The complete workflow includes:
- Dataset loading from structured directories
- Exploratory visualization of class distribution
- Data splitting into train/validation/test sets
- Real-time data augmentation using `ImageDataGenerator`
- Building and training a model on **GPU**
- Performance evaluation with confusion matrix and metrics
- Inference through a custom prediction function with visualization

---

## 🧬 Dataset Overview

- Structured directory of images per class
- Preprocessed using TensorFlow's `ImageDataGenerator`
- Image size standardized to **299x299** pixels
- Train: 81.7%, Validation: 9.1%, Test: 9.2%

  ![Brain Tumor Dataset](https://drive.google.com/uc?export=view&id=1P6t6IkzYGnLiCHAeTGFtA8daS-HZX4ba)


---

## 🧠 Model Architecture

- **Base Model**: Xception (pretrained on ImageNet, `include_top=False`)
- **Head**:
  - Flatten Layer
  - Dense(128, relu) + Dropout(0.25)
  - Dense(4, softmax)

**Loss**: Categorical Crossentropy  
**Optimizer**: Adamax  
**Metrics**: Accuracy, Precision, Recall

---

## 📈 Results & Evaluation

- Trained over **10 epochs** using GPU
- Tracked metrics: Accuracy, Precision, Recall, Loss
- Plotted learning curves across epochs
- Generated **confusion matrix** for final test set predictions

---

## 🔍 Prediction Function

A custom `predict()` function:
- Takes an MRI image path
- Resizes and normalizes the image
- Uses the trained model for inference
- Displays predicted class probabilities with a bar chart

```python
predict("/path/to/image.jpg")
```
![Brain Tumor Prediction Output](https://drive.google.com/uc?export=view&id=1P6t6IkzYGnLiCHAeTGFtA8daS-HZX4ba)


---

## 📂 Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `BT_99_.ipynb` | Main notebook with full workflow |
| `brain_tumor_model_99.h5` | Saved model for deployment/inference |
| `predict()` | Function to test model on unseen images |
| Dataset folders | MRI brain scan images by class |

---

## 📊 Key Highlights

- ✅ Transfer learning with Xception
- ✅ Data augmentation for improved generalization
- ✅ Visualized training metrics and confusion matrix
- ✅ Precision-focused evaluation
- ✅ User-friendly prediction interface

---

## 🚀 Future Enhancements

- Add **Grad-CAM** for explainable AI visualizations
- Deploy with **Streamlit** for web access
- Extend to segmentation (tumor localization)

---
