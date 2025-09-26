# Ophthalmology_project
Deep Learning in Ophthalmology

# Age Estimation from Ultra-Wide Field Fundus Images

This project aims to estimate a person's age from ultra-wide field (UWF) fundus images using deep learning, focusing on the optic disc region. We also evaluate the age gap (actual age – predicted age) as a potential biomarker for retinal diseases.

## Project Overview

### 0. Baseline Model with MobileViT
- A MobileViT model was trained directly on the non-processed images (full UWF) to obtain a baseline performance.
- This serves as a comparison with models trained on cropped optic disc regions.

### 1. Dataset Preparation
- Downloaded the UWF fundus image dataset from [Figshare](https://springernature.figshare.com/articles/dataset/Open_ultrawidefield_fundus_image_dataset_with_disease_diagnosis_and_clinical_image_quality_assessment/26936446?file=49014559).
- Built metadata files linking image filenames with participants' ages and diagnoses.

### 2. Optic Disc Cropping
- Instead of YOLO-based automatic detection, the optic disc regions were cropped manually to ensure precise localization.
- Output: standardized disc-centered images used for age prediction.

### 3. Age Prediction Model
- A deep learning model (MobileViT / ViT) was trained on the cropped images of healthy individuals.
- 5-fold cross-validation was used to assess generalization.
- Task: Predict chronological age as a regression problem.

### 4. Age Gap Analysis for Diseases
- The trained model was applied to disease images.
- The age gap = actual age – predicted age was computed.
- This metric is investigated as a potential biomarker of retinal disease.


---

## Notes
- Optic disc cropping was performed manually instead of YOLO training.
- Figures are included to document dataset distribution, prediction errors, and preprocessing workflow.
