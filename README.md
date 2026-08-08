# 🩺 EffNet-SVM: Diabetic Retinopathy Detection & Explainable AI Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://effnet-svm-dr.streamlit.app/)

An advanced medical decision support tool that utilizes a hybrid **EfficientNetV2-S + Support Vector Machine (SVM)** model to classify and explain the presence of **Diabetic Retinopathy (DR)** from retinal fundus photographs. 

> 🔗 **Live Web Application**: Access the live clinical workstation console here: [effnet-svm-dr.streamlit.app](https://effnet-svm-dr.streamlit.app/)

The application is built with a modular, scalable architecture and features dual-layered **Explainable AI (XAI)** utilizing both **Grad-CAM** and **LIME** to build trust and provide visual interpretability for clinicians.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Technology Stack](#-technology-stack)
4. [Folder Structure](#-folder-structure)
5. [Getting Started & Usage](#-getting-started--usage)
6. [Explainable AI (XAI) Methods](#-explainable-ai-xai-methods)
7. [GitHub Upload Instructions](#-github-upload-instructions)

---

## 🔍 Project Overview

Diabetic Retinopathy (DR) is a major complication of diabetes that causes damage to the blood vessels of the light-sensitive tissue at the back of the retina. It is a leading cause of blindness worldwide. Early detection through regular retinal screenings is critical to prevent permanent vision loss.

This project implements a hybrid machine learning pipeline trained on the **APTOS 2019 Blindness Detection** dataset. By combining the feature representation capabilities of deep learning (EfficientNetV2) with the robust boundary-finding properties of Support Vector Machines (SVM), the system achieves highly accurate binary classification (Diabetic Retinopathy vs. Normal).

---

## 🏗️ System Architecture

The pipeline consists of four distinct processing stages:

```
[Raw Fundus Image] 
       │
       ▼
[Ben Graham Preprocessing] ──► Normalizes illumination, highlights microaneurysms/hemorrhages
       │
       ▼
[EfficientNetV2-S]         ──► Extracts high-dimensional deep feature maps
       │
       ▼
[StandardScaler + SVM]     ──► Performs binary classification & outputs class probabilities
       │
       ▼
┌──────┴────────────────────────┐
│                               │
▼                               ▼
[Grad-CAM Saliency Map]  [LIME Superpixel Regions]
```

1. **Preprocessing (Ben Graham's Method)**: Fundus photographs often suffer from varied lighting and resolution. We subtract local Gaussian blur and normalize colors to enhance structural lesions (exudates, microaneurysms, hemorrhages).
2. **Deep Feature Extraction**: We pass the preprocessed image through a fine-tuned **EfficientNetV2-S** model and extract the features just before the final classification head (resulting in a 1280-dimensional feature vector).
3. **SVM Classification**: The feature vector is scaled using a trained `StandardScaler` and classified using a Support Vector Machine (`SVC` with probability estimation enabled) to output the diagnosis and confidence level.
4. **Visual Interpretability (XAI)**:
   * **Grad-CAM** reveals global activation maps showing what regions of the network's convolutional layers responded strongest.
   * **LIME** perturbs local superpixels to show which pixel groups positively drove the classifier's output.

---

## 💻 Technology Stack

* **Programming Language**: Python 3.10
* **Deep Learning Engine**: PyTorch (`torch`, `torchvision`)
* **Machine Learning & Core Math**: Scikit-Learn (`scikit-learn`, `joblib`), NumPy
* **Image Processing & Computer Vision**: OpenCV (`opencv-python`), Pillow (`pillow`), Scikit-Image (`scikit-image`)
* **Explainability Framework**: LIME (`lime`)
* **Web UI Dashboard**: Streamlit (`streamlit`)

---

## 📁 Folder Structure

The project has been structured systematically to separate the core ML models, preprocessing pipelines, and user interface elements:

```
EffNet_SVM_DR/
│
├── app.py                      # Main Streamlit UI Entrypoint (layout & widgets)
├── requirements.txt            # Project dependencies & versions
├── README.md                   # Complete project guide and details
│
├── src/                        # Source package for core functions
│   ├── __init__.py             # Marks the directory as a Python package
│   ├── preprocessing.py        # Image transformations & Ben Graham preprocessing
│   └── model_inference.py      # PyTorch Grad-CAM, LIME wrapper, & model loading
│
└── model/                      # Serialized ML assets (Git LFS recommended)
    ├── best_effnet_binary.pth  # Fine-tuned EfficientNetV2-S weights (PyTorch)
    ├── scaler_binary.pkl       # Trained StandardScaler (Joblib)
    └── svm_binary.pkl          # Trained Support Vector Machine classifier (Joblib)
```

---

## 🚀 Getting Started & Usage

### 1. Prerequisites
Ensure you have Python 3.10 installed on your system.

### 2. Environment Setup
Clone this project or navigate to its directory, then set up your virtual environment:

```powershell
# Create virtual environment if not already done
python -m venv venv310

# Activate the virtual environment
# On Windows (PowerShell):
.\venv310\Scripts\Activate.ps1
# On Linux/macOS:
source venv310/bin/activate
```

### 3. Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Running the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
This will start the local server and output a URL (usually `http://localhost:8501`) that you can open in any browser.

---

## 🔍 Explainable AI (XAI) Methods

Medical AI systems must be transparent. The dashboard contains two distinct XAI tabs:

### 1. Grad-CAM (Saliency Map)
Calculates gradients relative to the final convolutional layer of EfficientNetV2-S. It highlights which sections of the retina are drawing the most attention from the neural network.
* **Red/Yellow regions**: Indicate areas of strong network attention (typically microaneurysms or bleeding).
* **Blue regions**: Represent background structures that did not impact the classification decision.

### 2. LIME (Local Explanations)
A model-agnostic technique that segments the image into homogeneous patches (superpixels), randomly deactivates some of them, and observes the changes in classification confidence.
* Superpixels with a **yellow border** show the exact structures that contributed most positively to the final prediction (such as hard exudates or flame hemorrhages).

---

## 📤 GitHub Upload Instructions

Since the model contains large files (especially `best_effnet_binary.pth`, which is ~81 MB), we recommend using **Git Large File Storage (LFS)** or committing standard code files and uploading weights separately.

Here are the step-by-step terminal instructions to upload this project to GitHub:

### Step 1: Install Git LFS (Recommended for Large Files)
Download and install Git LFS from [git-lfs.com](https://git-lfs.github.com/) or via your package manager, then run:
```bash
git lfs install
```

### Step 2: Initialize Git & Track Large Files
Navigate to the `EffNet_SVM_DR` directory and initialize Git:
```bash
# Initialize local repo
git init

# Configure LFS to track PyTorch model weight files (.pth)
git lfs track "model/*.pth"
git lfs track "model/*.pkl"

# Add tracking files
git add .gitattributes
```

### Step 3: Create `.gitignore`
To prevent uploading virtual environments or cached runtime files, create a `.gitignore` file at the root:
```bash
# .gitignore
venv310/
__pycache__/
.streamlit/
.ipynb_checkpoints/
*.log
```

### Step 4: Commit Your Code
```bash
git add .
git commit -m "Initial commit: Modularized EffNet-SVM Diabetic Retinopathy Application with custom UI and XAI tools"
```

### Step 5: Push to GitHub
Go to [github.com](https://github.com), log in, create a new repository (e.g., `EffNet-SVM-DR-Detection`), and run:
```bash
# Link local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# Rename main branch
git branch -M main

# Push the code to GitHub
git push -u origin main
```

---

## 📬 Secure Contact & Feedback System Setup

The workstation includes a contact form in the sidebar that allows users to send messages directly to your email without exposing your personal email address in the codebase or UI. 

This relies on **Web3Forms** and Streamlit's secure secrets management:

1. **Get an Access Key**: Go to [web3forms.com](https://web3forms.com/), enter your email, and receive your free public access token in your inbox.
2. **Local Configuration**: Create a file named `.streamlit/secrets.toml` at the root of the project:
   ```toml
   web3forms_access_key = "your-web3forms-access-key-here"
   ```
   *(Note: `.streamlit/*secrets.toml` is already added to `.gitignore` to prevent committing your key to public repositories).*
3. **Cloud Deployment Configuration**:
   * Go to your **Streamlit Community Cloud Dashboard**.
   * Click **Settings** next to your app.
   * Navigate to **Secrets** and paste the same configuration snippet there.

