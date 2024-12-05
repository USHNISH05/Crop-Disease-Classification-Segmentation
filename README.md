CNN-ViT Classification Project
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


U-Net Image Segmentation Project
Overview
This project implements a U-Net architecture for semantic image segmentation using TensorFlow and Keras. The model is trained on a leaf disease segmentation dataset, where the goal is to segment diseased regions from input images. The project includes a custom data generator for efficiently loading images and masks and supports extensive training for precise segmentation.

Key Features
Custom U-Net Model: A fully convolutional U-Net architecture with skip connections for detailed segmentation.
Custom Data Generator: Handles on-the-fly loading and preprocessing of images and masks, ensuring efficient training.
Normalization and Augmentation: Input images are normalized, and mask values are processed for binary classification.
Training Flexibility: Configurable batch size, image size, and number of epochs.
Model Architecture
The U-Net model includes:

Encoder Path:
Convolutional layers followed by dropout for feature extraction.
Max pooling for spatial down-sampling.
Decoder Path:
Up-sampling layers for spatial reconstruction.
Skip connections for incorporating low-level features from the encoder.
Output Layer:
A Conv2D layer with a sigmoid activation for pixel-wise binary classification.
Dataset Preparation
The dataset consists of:

Images: Stored in .jpg format.
Masks: Stored in .png format and loaded in grayscale mode for binary segmentation.
The custom data generator:

Loads and preprocesses images and masks.
Normalizes images to [0, 1] and converts masks to binary format (mapping pixel value 38 to 1 and others to 0).
Ensures correspondence between images and masks using consistent file ordering.
Training Pipeline
Model Compilation:

Optimizer: Adam with a learning rate of 1e-4.
Loss Function: Binary Crossentropy for binary segmentation.
Metrics: Accuracy to monitor performance.
Training:

Data is fed using the custom data generator.
Model is trained for 500 epochs to achieve high segmentation accuracy.
Model Saving:

The trained model is saved as unet_500.keras.
Code Structure
Custom Data Generator:
Efficiently loads images and masks from directories.
Ensures data consistency and performs preprocessing.
U-Net Model:
Defined with an input shape of (256, 256, 3) and binary segmentation output.
Training:
Compiles the model with Adam optimizer and binary crossentropy loss.
Trains the model using the data generator for the specified number of epochs.
How to Use
Clone this repository and install the required dependencies (TensorFlow).
Update the image_dir and mask_dir paths to point to your dataset.
Run the script to start training the U-Net model.
Use the saved model for inference or further fine-tuning.
Requirements
Python 3.7+
TensorFlow 2.5+
GPU (optional, recommended for faster training)
Results
The U-Net model demonstrates strong segmentation capabilities, effectively isolating diseased regions in leaf images. Training for 500 epochs ensures robust learning and high accuracy.

References
U-Net Architecture
Kaggle Leaf Disease Segmentation Dataset


