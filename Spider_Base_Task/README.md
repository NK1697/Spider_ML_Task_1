# Base Task – Fashion MNIST Classification

This project implements a deep learning model for classifying Fashion-MNIST images using a fully connected neural network (MLP).

##  Dataset

- Fashion-MNIST
- 10 classes of clothing items
- 28x28 grayscale images

## Model Architecture

- Fully Connected Neural Network (MLP)
- Input layer: 784 neurons (flattened image)
- Hidden layers: multiple dense layers with ReLU activation
- Output layer: 10 classes (Softmax)

## Workflow

1. Load dataset
2. Normalize images
3. Flatten input
4. Train MLP model
5. Evaluate accuracy
6. Generate predictions

## Evaluation

- Accuracy used as primary metric
- Training vs validation performance tracked
- Model generalization evaluated on test set

## Output

- Trained model saved in `saved_models/`
- Submission file generated as `submission.csv`

## Key Learnings

- Image classification using neural networks
- Importance of feature representation
- Overfitting vs generalization
