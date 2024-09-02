# Breast Cancer Histology Images Classification

## Project Overview
This project classifies breast cancer histology images into four categories: **Benign**, **InSitu**, **Invasive**, and **Normal**. The classification is performed using a Global-Local Neural Network (GLNET) model built with TensorFlow and Keras.

## Directory Structure
The project is divided into the following files:

- **`data.py`**: Handles data loading, preprocessing, and patch extraction.
- **`model.py`**: Contains the GLNET model architecture and functions for training and evaluating the model.
- **`predict.py`**: Provides functions for predicting classes of new images and plotting the results.
- **`main.py`**: The entry point for running the project, orchestrating data loading, model training, evaluation, and prediction.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.7+ installed. Install the required Python packages by running:

```bash
pip install -r requirements.txt
