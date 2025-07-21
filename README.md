# Ophthalmology_project
Deep Learning in Ophthalmology.

# Age Estimation from Ultra-Wide Field Fundus Images

This project aims to estimate a person's age from ultra-wide field (UWF) fundus images using deep learning, focusing on the optic disc region. We also evaluate the **age gap** (actual age – predicted age) as a potential biomarker for retinal diseases.

## Project Overview

The project is structured into several key stages:

### 1. Dataset Preparation

- Download a UWF fundus image dataset.
- Create a manual **bounding box annotation** for 600 images to localize the optic disc.

> A dedicated annotation script is provided to assist with bounding box creation.

### 2. Optic Disc Detection and Cropping

- Train a YOLO model (e.g., YOLOv5) using the 600 annotated images.
- Apply the trained YOLO model to crop the optic disc region from all images.
- Output: standardized disc-centered images for age prediction.

### 3. Age Prediction Model

- Train a deep learning model (e.g., **MobileViT** or **ViT**) on the cropped images of healthy individuals.
- Use **5-fold cross-validation** to evaluate generalization.
- Target: Predict chronological age as a regression task.


