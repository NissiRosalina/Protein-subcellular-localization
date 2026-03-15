# Protein Subcellular Localization Classifier
A deep learning model that classifies proteins into 19 subcellular localization categories using 4-channel fluorescence microscopy images from the Human Protein Atlas (HPA) Kaggle Competition.

## Overview
Proteins must be in the right place inside a cell to function correctly. Mislocalized proteins are linked to diseases like cancer and neurodegeneration. Manually annotating where proteins are located under a microscope is slow and expensive — this project automates that process using a convolutional neural network.
The model takes in 4-channel fluorescence images (red, green, blue, yellow) and predicts which of 19 subcellular compartments the protein localizes to. Since proteins can appear in multiple locations simultaneously, this is a multi-label classification problem.

## Dataset
HPA Single Cell Image Classification — Kaggle  
Each sample contains 4 grayscale PNG images, one per fluorescence channel:
Channel	Stain	Target
Red	Microtubules	Cytoskeleton structure
Green	Protein	Protein of interest
Blue	DAPI	Nucleus
Yellow	ER	Endoplasmic reticulum
19 localization classes (e.g. nucleus, cytosol, mitochondria, plasma membrane)
Heavily imbalanced — some classes are much rarer than others


## Model architecture
Backbone: EfficientNet-B0 (pretrained on ImageNet via timm)
Input: Modified first conv layer — accepts 4 channels instead of the standard 3
Output: 19-dimensional sigmoid output (one probability per class)
Loss: "BCEWithLogitsLoss" with class frequency weights to handle imbalance
Optimizer: Adam (lr = 3e-4)
Metric: Sample-averaged F1 score

### Why EfficientNet?
EfficientNet achieves strong accuracy with fewer parameters than heavier architectures like ResNet-50. This makes it practical for training on Kaggle's free GPU tier within time limits.

### Why 4 channels?
Standard pretrained models expect 3-channel (RGB) images. Here, the 4th channel (yellow/ER stain) adds biologically meaningful information. The pretrained weights for the first 3 channels are preserved; the 4th channel is initialized as their average — a common and effective technique for extending pretrained models to new modalities.

## Results
The model achieved a best validation F1 score of **0.4418** at epoch 4 out of 5, 
trained on a 3,000 sample subset of the full HPA dataset. Loss consistently 
decreased across epochs, suggesting the model was still improving — 
training on the full dataset would likely push F1 significantly higher.

## Project Structure
```
Protein-subcellular-localization/
│
├── protein_classifier.py   # pipeline: dataset, model, training, evaluation
├── requirements.txt        # Python dependencies
└── README.md
```


## How to run
This project is designed to run on Kaggle Notebooks where the HPA dataset is directly available.
Go to Kaggle and create a new notebook
Add the dataset: 'hpa-single-cell-image-classification'
Enable GPU in notebook settings
Upload and run 'protein_classifier.py'


## Requirements
torch
torchvision
timm
numpy
pandas
Pillow
scikit-learn
tqdm
matplotlib

## Key concepts used
Transfer learning with pretrained CNNs
Multi-label classification with binary cross-entropy
Class imbalance handling via weighted loss
Custom PyTorch Dataset and DataLoader
Data augmentation (horizontal/vertical flips)
Model checkpointing (saves best F1 model)

