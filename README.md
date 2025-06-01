# Pokémon Character Detector

A deep learning-powered system for detecting and classifying Pokémon characters from images using computer vision and convolutional neural networks.

## Project Overview

This project implements an intelligent Pokémon recognition system that can accurately identify and classify Pokémon characters from the first generation. The system combines advanced computer vision techniques with deep learning to achieve high accuracy in character recognition.

### Key Objectives

- **Data Collection**: Automated web scraping for diverse Pokémon image datasets
- **Preprocessing Pipeline**: Comprehensive data preprocessing and augmentation
- **Model Design**: Efficient CNN architecture for Pokémon classification  
- **Performance Optimization**: Model evaluation and accuracy optimization
- **Deployment Ready**: Easily deployable Pokémon recognition system

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- PyTorch 1.8 or higher
- TorchVision
- Additional dependencies (see `requirements.txt`)

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google Drive (for Colab users)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Data Pipeline

### Data Collection Strategy

The project employs automated web scraping to gather comprehensive image datasets:

- **Sources**: Google Images and Bing using icrawler
- **Coverage**: 20 first-generation Pokémon characters
- **Organization**: Structured directory hierarchy for easy management

### Preprocessing & Augmentation

All images undergo standardized preprocessing with extensive augmentation:

- **Standardization**: Resize to 224×224 pixels
- **Augmentation Techniques**:
  - Rotation transformations (-30° to 30°)
  - Horizontal flipping
  - Brightness and contrast adjustments
  - Color saturation modifications
  - Gaussian blur effects
  - Sharpness enhancements
  - Random cropping and resizing
  - Noise injection
  - Perspective transformations

### Dataset Distribution

- **Training Set**: 70% of total data
- **Validation Set**: 15% of total data  
- **Test Set**: 15% of total data

## Model Architecture

The classification system utilizes a sophisticated CNN architecture:

- **Convolutional Layers**: Multiple layers with ReLU activation functions
- **Pooling Strategy**: Max pooling for effective dimensionality reduction
- **Dense Layers**: Fully connected layers for final classification
- **Regularization**: Dropout layers to prevent overfitting
- **Optimization**: Adam optimizer with cross-entropy loss

## Usage

### Training the Model

```bash
python train.py --data_dir /path/to/dataset --epochs 50 --batch_size 32
```

### Making Predictions

```python
from predictor import predict_pokemon

# Predict Pokémon from image
result = predict_pokemon('pikachu.jpg')
print(f"Predicted Pokémon: {result['class']} with {result['confidence']:.2f} confidence")
```

### Expected Output

```
Predicted Pokémon: Pikachu with 0.97 confidence
```

## Performance Metrics

Model evaluation results on the test dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| **Precision** | 0.93 |
| **Recall** | 0.92 |
| **F1 Score** | 0.925 |
| **Top-3 Accuracy** | 98.2% |

## Key Features

- **Automated Pipeline**: End-to-end data collection and processing
- **Robust Augmentation**: Comprehensive image transformation techniques
- **Optimized Architecture**: Efficient CNN design for fast inference
- **Detailed Analytics**: Comprehensive performance evaluation metrics
- **User-Friendly Interface**: Simple prediction API for easy integration

## Future Roadmap

- **Extended Coverage**: Expand to include more Pokémon generations
- **Real-Time Detection**: Implement live video stream processing
- **Mobile Optimization**: Cross-platform mobile application development
- **Enhanced Accuracy**: Improved classification for visually similar Pokémon
- **Web Interface**: User-friendly web application for broader accessibility

