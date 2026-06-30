# TensorFlow-CNN-Image-Classification

A deep learning project that uses Convolutional Neural Networks (CNNs) to detect manufacturing defects in industrial cables, simulating a real-world quality control system for industrial automation companies like ABB.

The project includes two implementations:
1. **General CNN benchmark** trained on CIFAR-10 for baseline image classification
2. **Industrial defect detection model** trained on the MVTec Anomaly Detection (MVTec AD) dataset, applied to cable quality inspection

---

## Project Overview

Quality control is critical in industrial manufacturing — companies like ABB rely on automated visual inspection systems to catch defective components (cables, connectors, electronics) before they reach the assembly line. This project builds a CNN-based binary classifier that distinguishes **good** cables from **defective** ones, using real defect categories such as bent wires, cut insulation, and missing components.

---

## Datasets

### 1. CIFAR-10 (Baseline)
- 60,000 32x32 color images across 10 general object classes
- Used to validate the CNN architecture on a standard benchmark before applying it to a domain-specific problem

### 2. MVTec Anomaly Detection — Cable Category
- Industry-standard dataset for unsupervised anomaly detection, published by MVTec Software GmbH
- Real photographs of cables with the following defect types:
  - Bent wire
  - Cable swap
  - Cut inner/outer insulation
  - Missing cable / missing wire
  - Poke insulation
- Source: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## Model Architecture

A custom CNN built with TensorFlow/Keras:

```
Input (128x128x3)
 → Conv2D(32) → MaxPooling2D → Dropout(0.25)
 → Conv2D(64) → MaxPooling2D → Dropout(0.25)
 → Conv2D(128) → MaxPooling2D → Dropout(0.25)
 → Flatten
 → Dense(128, ReLU) → Dropout(0.5)
 → Dense(1, Sigmoid)
```

- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** Binary classification (Good vs Defective)

Dropout layers were added at each stage specifically to address overfitting — an earlier version of the model without dropout achieved 100% training accuracy but only 38% test accuracy. Adding dropout and balancing the training set with defective samples improved test accuracy to **78%**.

---

## Results

| Model Version              | Dataset    | Test Accuracy |
|----------------------------|------------|----------------|
| Baseline CNN                | CIFAR-10   | 70.47%         |
| Cable Defect Detection CNN  | MVTec AD   | 78%            |

### Key Learning: Overfitting Fix
The initial cable defect model was trained only on "good" cable images, causing it to fail on defective samples at test time (38% accuracy, with training accuracy at 100%). The fix involved:
- Including defective images in the training set (not just test set)
- Adding Dropout layers after each convolutional block
- Balancing the train/test split across all defect categories

This took accuracy from 38% → 78%.

---

## Project Structure

```
TensorFlow-CNN-Image-Classification/
│
├── main.py                  # Baseline CNN on CIFAR-10
├── mvtec_cnn.py              # Cable defect detection CNN (MVTec AD)
├── data/                     # MVTec AD dataset (gitignored)
├── abb_cnn_model.keras       # Saved CIFAR-10 model
├── abb_cable_defect_model.keras  # Saved cable defect model
├── training_results.png      # CIFAR-10 accuracy/loss plots
├── mvtec_training_results.png # Cable defect accuracy/loss plots
├── predictions.png           # CIFAR-10 sample predictions
├── mvtec_predictions.png     # Cable defect sample predictions
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Set up environment**
```bash
python -m venv tf_env
tf_env\Scripts\activate      # Windows
pip install tensorflow matplotlib numpy pandas scikit-learn pillow
```

2. **Run baseline CIFAR-10 model**
```bash
python main.py
```

3. **Run cable defect detection model**
   - Download the MVTec AD cable dataset from the [official site](https://www.mvtec.com/company/research/datasets/mvtec-ad)
   - Extract it into `data/`
   - Run:
```bash
python mvtec_cnn.py
```

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy, Pillow
- Matplotlib (visualization)
- Scikit-learn (data utilities)

---

## Relevance to Industrial Automation

This project mirrors real defect-detection pipelines used in manufacturing quality assurance, particularly relevant to industrial automation and robotics companies. The same CNN-based approach generalizes to other components — PCBs, metal parts, fasteners — using the same MVTec AD dataset categories.

---

## Author

Yash Gupta
GitHub: [YashGupta018](https://github.com/YashGupta018)
