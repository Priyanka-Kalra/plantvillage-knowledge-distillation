# PlantVillage Knowledge Distillation & Model Diversity

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen.svg)](https://github.com/spMohanty/PlantVillage-Dataset)
[![Status](https://img.shields.io/badge/Status-Research%20Manuscript-success.svg)](./Paper_Manuscript.pdf)

> **Associated Research Manuscript:**  
> *"Code Diversity in Model Training: A Comparative Study for Dataset Accuracy Enhancement in Plant Disease Detection"*  
> **Authors:** Ravinder M., Priyanka Kalra, Sunishtha Nangal, Jafrin Khan, Yashica Dhalla  
> **Affiliation:** Department of Computer Science & Engineering, Indira Gandhi Delhi Technical University for Women (IGDTUW), New Delhi, India  
> **Target Journal:** *Journal of Ambient Intelligence and Humanized Computing* (AIHC)

---

## 📅 Original Project Timeline

| Date / Period | Milestone / Activity | Corresponding Code |
| :--- | :--- | :--- |
| **December 2023** | Baseline model setup & initial dataset pipeline exploration | [`Original/`](./Original/) |
| **February 2024** | Exploration of teacher-student combinations, dropout tuning, & ResNet50 distillation | [`Comb5/`](./Comb5/), [`Comb6/`](./Comb6/), [`Comb1/`](./Comb1/) |
| **March 2024** | Self-distillation analysis and initial multi-student ensemble experiments | [`SelfDistillation/`](./SelfDistillation/), [`EnsembleMethod/`](./EnsembleMethod/) |
| **April 2024** | Advanced ensemble distillation with multi-student aggregation & thesis compilation | [`EnsembleMethod2/`](./EnsembleMethod2/) |
| **September 2024 – 2025** | Final experimental consolidation & research paper manuscript preparation | [`Paper_Manuscript.pdf`](./Paper_Manuscript.pdf) |

---

## 📌 Project Overview

Automating plant disease diagnosis from leaf imagery is crucial for sustainable agriculture and food security. While deep Convolutional Neural Networks (CNNs) achieve high classification accuracy, deployability on edge agricultural devices (smartphones, handheld field scanners, IoT drones) is constrained by memory, compute limits, and dataset limitations (scarcity, class imbalance, and background noise).

This project investigates **Knowledge Distillation (KD)**, **Self-Distillation**, and **Ensemble Distillation** techniques across 38 distinct crop-disease classes from the **PlantVillage** dataset:
1. **Model Compression:** Compressing deep, over-parameterized teacher networks into lightweight student models suitable for edge inference.
2. **Transfer of Soft Dark Knowledge:** Leveraging teacher class-probability distributions with temperature scaling to regularize the student network.
3. **Overcoming Overfitting:** Comparing standard CNN teachers against pre-trained backbones (**ResNet50**) and multi-student ensembles to prevent over-memorization of training samples.

---

## 📊 Experimental Results & Model Comparison

Across five systematic distillation methodologies, model capacity, regularization, and teacher architectures were progressively refined:

| Implementation | Strategy | Teacher Model (Params) | Student Model (Params) | Teacher Train / Val Acc | Distilled Student Train / Val Acc | Student from Scratch (Train Acc) | Distillation Loss | Key Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Impl. 1** ([`SelfDistillation`](./SelfDistillation/)) | Self-Distillation | Custom CNN (1.25M) | Same Model (1.25M) | 89.67% / 86.41% | 63.28% / 57.74% | 90.59% (87.76% val) | 1.74e-4 | Self-distillation struggled to extract novel feature representations from its own predictions. |
| **Impl. 2** ([`Comb6`](./Comb6/)) | High-Capacity Teacher KD | 29-Layer CNN (52.75M) | 10-Layer CNN (1.25M) | 98.08% / 26.96% | 85.04% / 82.91% | 87.05% (87.05% val) | 0.0828 | Complex custom teacher severely overfitted on training data (val acc 26.96%), limiting distillation transfer. |
| **Impl. 3** ([`Comb1`](./Comb1/)) | Regularized Teacher KD | 29-Layer CNN + High Dropout (52.75M) | 10-Layer CNN (1.25M) | 96.43% / 11.05% | 84.75% / 82.74% | 70.53% | 0.1069 | Increased teacher dropout did not resolve feature overfitting; student still outperformed teacher on validation. |
| **Impl. 4** ([`Original`](./Original/)) | Balanced CNN Architecture | 23-Layer CNN (23.66M) | 11-Layer CNN (4.31M) | 98.58% / 83.15% | 90.65% / 83.20% | 67.86% | 0.0691 | Balanced capacity improved stability; distilled student achieved 90.65% training accuracy. |
| **Impl. 5** ([`Comb5`](./Comb5/) - **Best**) | **Pre-trained ResNet50 Transfer KD** | **ResNet50 Backbone (23.66M)** | **Compact CNN (4.31M)** | **98.58% / 83.15%** | **94.72% / 87.87%** | **81.23%** | **0.0208** | **Best Performance:** Reduced validation gap to 3.86%, achieved **87.87%** validation accuracy with ~82% parameter reduction over the teacher. |
| **Ensembles** ([`EnsembleMethod`](./EnsembleMethod/), [`EnsembleMethod2`](./EnsembleMethod2/)) | Multi-Student Ensemble KD | Custom Teacher | Multi-Student Aggregation | 97.79% / 28.90% | Aggregated student predictions | — | Joint loss | Evaluated ensemble knowledge aggregation across multiple heterogeneous student sub-networks. |

### 🏆 Key Takeaways:
- **ResNet50 as Teacher (Impl. 5)** yielded the highest distillation quality, transferring robust features to the compact student model (**94.72%** train / **87.87%** validation accuracy).
- The distilled student achieved **superior generalization** compared to training the same architecture from scratch (which achieved only **81.23%** training accuracy).
- High parameter reduction (~82%) makes the student model significantly lighter and faster for real-time mobile crop disease screening.

---

## 🗂️ Repository Directory Structure

```text
plantvillage-knowledge-distillation/
├── Paper_Manuscript.pdf         # Research paper manuscript submitted to Springer AIHC
├── README.md                    # Project documentation, results, & setup guide
├── requirements.txt             # Environment dependencies for reproducibility
├── .gitignore                   # Standard Python/Jupyter git exclusion rules
├── Comb5/                       # Implementation 5: ResNet50 teacher knowledge distillation (Top Performer)
│   ├── Plant_Disease_Detection.ipynb
│   ├── Plant_Disease_Detection 2.ipynb
│   └── label_transform.pkl
├── Comb6/                       # Implementation 2: Deep 29-layer CNN teacher to 10-layer student
│   ├── Plant_Disease_Detection.ipynb
│   └── label_transform.pkl
├── Comb1/                       # Implementation 3: Teacher regularization with increased dropout
│   ├── Plant_Disease_Detection.ipynb
│   └── label_transform.pkl
├── Original/                    # Implementation 4: Balanced CNN architecture distillation
│   └── Plant_Disease_Detection.ipynb
├── SelfDistillation/            # Implementation 1: Self-distillation experiment
│   ├── Plant_Disease_Detection.ipynb
│   └── label_transform.pkl
├── EnsembleMethod/              # Multi-student distillation ensemble pipeline
│   └── ensemblecode.ipynb
└── EnsembleMethod2/             # Refined multi-student ensemble distillation
    └── ensemblecode.ipynb
```

---

## 🌿 Dataset Details

- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Images:** 48,888 leaf images across **38 classes** (including healthy and diseased varieties of Tomato, Potato, Apple, Corn, Grape, Pepper, etc.).
- **Dataset Split:**
  - **Training Set:** 43,429 images (~89%)
  - **Validation Set:** 5,459 images (~11%)
- **Preprocessing:**
  - Normalized with `rescale=1./255` to scale pixel intensities to `[0, 1]`.
  - Target input resolution: $256 \times 256 \times 3$ (or $224 \times 224 \times 3$ for pre-trained backbones).
  - Real-time data augmentation using `ImageDataGenerator` (random rotations, zoom, shear, and horizontal flips) to minimize overfitting.

---

## ⚙️ Instructions for Reproducing Experiments

### 1. Prerequisites
- **Python:** 3.8 to 3.11
- **Hardware:** GPU with $\ge 8\text{ GB}$ VRAM recommended (experiments were executed on an 8-core GPU environment).

### 2. Environment Setup

Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/Priyanka-Kalra/plantvillage-knowledge-distillation.git
cd plantvillage-knowledge-distillation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Download the [PlantVillage dataset from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Organize your dataset directory structure:
   ```text
   dataset/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Tomato___Late_blight/
   │   └── ...
   └── val/
       ├── Apple___Apple_scab/
       ├── Tomato___Late_blight/
       └── ...
   ```
3. Update the `train_dir` and `val_dir` paths in the first code cell of the notebook you wish to run.

### 4. Running the Notebooks
Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

- **To run the best-performing distillation model (ResNet50 $\rightarrow$ Compact Student):**  
  Open and execute [`Comb5/Plant_Disease_Detection.ipynb`](./Comb5/Plant_Disease_Detection.ipynb).
- **To run the self-distillation experiment:**  
  Open [`SelfDistillation/Plant_Disease_Detection.ipynb`](./SelfDistillation/Plant_Disease_Detection.ipynb).
- **To run the multi-student ensemble distillation:**  
  Open [`EnsembleMethod/ensemblecode.ipynb`](./EnsembleMethod/ensemblecode.ipynb).

---

## 📖 Citation

If you use this codebase or refer to the findings in your research, please cite:

```bibtex
@article{kalra2024codediversity,
  title={Code Diversity in Model Training: A Comparative Study for Dataset Accuracy Enhancement in Plant Disease Detection},
  author={Ravinder, M. and Kalra, Priyanka and Nangal, Sunishtha and Khan, Jafrin and Dhalla, Yashica},
  journal={Journal of Ambient Intelligence and Humanized Computing},
  year={2024},
  note={Manuscript under review}
}
```
