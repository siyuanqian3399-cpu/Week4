# Week4
# Introduction
This project extends the notebook Chapter1_Unsupervised_Learning_Methods_Michel.ipynb and applies unsupervised learning to classify radar echo signals into lead (open water) and sea ice. For each class, the mean echo waveform and its standard deviation are computed to characterize the typical echo shape. The classification results are quantitatively evaluated against the ESA reference classification using a confusion matrix. This repository documents the complete workflow for data processing, classification, and evaluation, enabling reproducible analysis.
```sh
from google.colab import drive
drive.mount('/content/drive')
```
