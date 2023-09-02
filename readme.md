# Collaborative Segmentation Method Using Uncertainty-Guided Annotation Sampling
## Human-in-the-Loop Deep Learning for Digital Pathology

### Overview

This project introduces a novel human-in-the-loop deep learning pipeline for enhancing the efficiency and accuracy of cancer cell segmentation in digital pathology. 

- **Efficiency and Accuracy**: Utilizes per-pixel and local-level model uncertainty to guide the annotation process.
- **Informed Decision-making**: Provides pathologists with valuable insights into the model's limitations.
- **Out-of-Domain Handling**: Specifically designed to handle Out-of-Domain (OOD) scenarios commonly encountered in medical data.
## Data
[Camelyon](https://camelyon17.grand-challenge.org/)
In our study, we employed the Camelyon dataset, focusing on Whole Slide Images (WSIs) to perform pixel-level segmentation.
Training Data: Our model was trained on Camelyon 16, specifically using data obtained from Radboud University Medical Center.
Uncertainty Measurement: To evaluate the model's uncertainty, we utilized a diverse set of training and test sets sourced from various medical centers.
### Results

Our pipeline demonstrates that using fewer annotated samples can still yield higher segmentation performance. This is achieved by focusing on out-of-domain areas identified through model uncertainty measurements.

### Model and Training

The pipeline employs five ensemble models of an adapted version of nnUNet, specifically tuned for pathology data for both segmentation and uncertainty measurements.

- **Data**: Trained on Whole Slide Images (WSIs).
- **Uncertainty Measurements**: Incorporates uncertainty to identify and correct misclassifications.

> **Note**: This pipeline is flexible; you can substitute nnUNet with any other network that provides uncertainty measurements.

[nnUNet for Pathology](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v1)
![Overview](images/overview.PNG)

### Quickstart Guide

1. **Prepare Your Dataset**:  
   ```bash
   python3 create_training.py

2. **Train the ensembles of the segmentation network**:
   
   ```bash
   nnunet plan_train task_name  root --network 2d --planner3d None --planner2d ExperimentPlanner2D_v21_RGB_scaleTo_0_1_bs8_ps512 --plans nnUNet_RGB_scaleTo_0_1_bs8_ps512 --trainer nnUNetTrainerV2_BN --fold 0 to 4
   
 3. **Compute uncertainty on new domain**:
     ```bash
       python3 nnunet_inference.py folder taskname 

 4. **Creat data from the area that model is the most uncertain about**:
    ```bash
       python3 uncertainty_sampling.py
5. **Retrain the egmentation network with in-domain and out-of-domain samples**:
   ```bash
   nnunet plan_train task_name  root --network 2d --planner3d None --planner2d ExperimentPlanner2D_v21_RGB_scaleTo_0_1_bs8_ps512 --plans nnUNet_RGB_scaleTo_0_1_bs8_ps512 --trainer nnUNetTrainerV2_BN --fold 0
   
 6.. **Test**:
       python3 nnunet_inference.py folder taskname

    
## Quickstart guide
The framework is modular and adaptable. We recommend using nnUNet with five-folds for segmentation and applying your specific uncertainty measurements for maximum effectiveness.

