# Collaborative Segmentation Method Using Uncertainty-Guided Annotation Sampling
## Human-in-the-Loop Deep Learning for Digital Pathology

### Overview

This project introduces a novel human-in-the-loop deep learning pipeline for enhancing the efficiency and accuracy of cancer cell segmentation in digital pathology. 

- **Efficiency and Accuracy**: Utilizes per-pixel and local-level model uncertainty to guide the annotation process.
- **Informed Decision-making**: Provides pathologists with valuable insights into the model's limitations.
- **Out-of-Domain Handling**: Specifically designed to handle Out-of-Domain (OOD) scenarios commonly encountered in medical data.

### Results

Our pipeline demonstrates that using fewer annotated samples can still yield higher segmentation performance. This is achieved by focusing on out-of-domain areas identified through model uncertainty measurements.

### Model and Training

The pipeline employs five ensemble models of an adapted version of nnUNet, specifically tuned for pathology data for both segmentation and uncertainty measurements.

- **Data**: Trained on Whole Slide Images (WSIs).
- **Uncertainty Measurements**: Incorporates uncertainty to identify and correct misclassifications.

> **Note**: This pipeline is flexible; you can substitute nnUNet with any other network that provides uncertainty measurements.

[nnUNet for Pathology](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v1)

### Quickstart Guide

1. **Prepare Your Dataset**:  
   ```bash
   python3 create_training.py

![Overview](images/overview.PNG)



## Quickstart guide

The collaborative is based on certainty and is flexible to adapt it to the certainty of your choice. We suggest to use nnunet segmentation with five folds and apply on your 

