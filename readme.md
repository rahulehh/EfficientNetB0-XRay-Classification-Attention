# Chest X-ray Classification Models

This repository contains two deep learning models for classifying chest X-ray images into distinct categories: **COVID-19**, **NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS**. The goal is to build and evaluate models that can accurately classify chest X-ray images using a convolutional neural network (CNN) approach. 

## Models

### 1. **ChestXRayClassificationWithoutAttention**
This model utilizes the **EfficientNetB0** architecture as the base model for image classification without the incorporation of attention mechanisms. It includes the following steps:

- **Data Preprocessing**: Loads and organizes the X-ray images into a DataFrame with labels.
- **Dataset Splitting**: The dataset is split into training/validation and test sets with stratification to ensure class distribution is maintained.
- **Model Architecture**: Uses EfficientNetB0 as the base model and adds custom layers for classification.
- **K-Fold Cross Validation**: Implements 5-fold cross-validation to robustly evaluate the model’s performance and prevent overfitting.
- **Final Model Evaluation**: The best model is selected based on cross-validation performance and evaluated on the test set.
- **Misclassified Predictions**: Misclassified predictions are visualized to identify failure cases.
- **Grad-CAM Visualization**: Class activation maps are generated to interpret the regions of the X-ray images that the model focuses on.

### 2. **ChestXRayClassificationWithAttention**
This model extends the previous architecture by adding **attention mechanisms** to the EfficientNetB0 base model. This enables the model to focus on more relevant features within the images for improved performance. The process is similar to the previous model, with the following key differences:

- **Attention Mechanism**: Integrates attention layers into the base EfficientNetB0 architecture.
- **Augmented Data Generation**: Includes various augmentation techniques (rotation, shear, zoom, flips) to enhance the training dataset.
- **K-Fold Cross Validation**: Similar to the previous model, K-fold cross-validation is used for robust performance evaluation.
- **Grad-CAM Visualization**: Class activation maps are visualized to understand the model's decision-making process.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- Matplotlib
- OpenCV
- pandas
- numpy

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/chest-xray-classification.git
    cd chest-xray-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that your dataset is properly placed in the `/datasets` directory.

## Training the Models

1. **Without Attention**: 
    ```python
    python ChestXRayClassificationWithoutAttention.py
    ```

2. **With Attention**:
    ```python
    python ChestXRayClassificationWithAttention.py
    ```

## Results

- **K-Fold Cross Validation**: Both models are evaluated with 5-fold cross-validation to ensure generalization.
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices are reported for both models.
- **Grad-CAM Visualizations**: Activation maps are provided for each model to visualize critical image regions.

## Conclusion

These models aim to improve the accuracy and interpretability of chest X-ray classification. The addition of attention mechanisms in the second model is intended to improve feature extraction, which may lead to better performance in detecting the target classes. 
