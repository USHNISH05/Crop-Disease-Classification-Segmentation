Overview
This project demonstrates a hybrid approach to image classification by combining a convolutional neural network (CNN) backbone (MobileNetV3Small) with a Vision Transformer (ViT) architecture. The model is trained on the Plant Village 400 dataset to classify 38 different plant disease classes, utilizing TensorFlow and Keras for implementation.

Key Features
Dataset Preprocessing: Images are loaded using TensorFlow's image_dataset_from_directory, resized to 256x256, and normalized for optimal training performance.
MobileNetV3Small Backbone: Utilized as a feature extractor for efficient and lightweight convolutional feature generation.
Vision Transformer Integration: Includes transformer encoders with class tokens, positional embeddings, and multi-head self-attention for enhanced global context understanding.
Custom Layer - Class Token: Implements a learnable class token layer to align the input with transformer-based processing.
Callbacks: Early stopping and learning rate scheduling are integrated to optimize training.
Model Architecture
The model is designed as follows:

Base Feature Extractor: MobileNetV3Small pre-trained on ImageNet, frozen to retain generic feature extraction capabilities.
Transformer Encoder: Includes dense projection layers, class token addition, and multi-head self-attention layers.
Output Layer: A dense layer with softmax activation for 38-class classification.
Training Pipeline
Data Augmentation & Preprocessing:

Training and validation datasets are split with a validation ratio of 20%.
Image caching and prefetching are used for performance optimization.
Model Training:

Optimizer: Adam with a learning rate of 1e-4.
Loss Function: Sparse Categorical Crossentropy.
Metrics: Accuracy.
Callbacks: Early stopping (patience = 5) and learning rate reduction on plateau (factor = 0.5, patience = 3).
Evaluation:

Monitored using validation accuracy and loss metrics during training.
Code Structure
Dataset Preparation:

Data is loaded from the specified directory.
Normalization and batching are applied for efficient training.
Model Definition:

ClassToken layer for integrating class tokens.
Transformer encoder with self-attention and feed-forward layers.
Positional embeddings for sequence representation.
Training and Callbacks:

Early stopping and learning rate scheduling are implemented to prevent overfitting.
Model Saving:

Final model is saved as cnn_vit_98.keras.
How to Use
Clone this repository and ensure TensorFlow is installed.
Update the dataset directory path (dataset_dir) to point to your local dataset.
Run the script to train the model.
Evaluate the saved model or fine-tune it as needed.
Requirements
Python 3.7+
TensorFlow 2.5+
NVIDIA GPU with CUDA support (optional for faster training)
Results
The model achieves high classification accuracy by leveraging the strengths of CNNs for feature extraction and ViTs for capturing global relationships.

References
MobileNetV3
Vision Transformer (ViT)
Plant Village 400 Dataset





