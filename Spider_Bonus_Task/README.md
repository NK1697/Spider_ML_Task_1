# Bonus Task – Autoencoder on Fashion-MNIST

This project implements a fully connected (MLP-based) autoencoder trained on the Fashion-MNIST dataset. The goal is to learn a compressed latent representation of images and reconstruct them back as accurately as possible.

## Objective

- Compress 28×28 images into a lower-dimensional latent space (bottleneck)
- Reconstruct original images from compressed representation
- Analyze reconstruction quality and information loss

## Model Architecture
### Encoder
- Input: 784-dimensional vector (flattened image)
- Fully connected layers
- Output: latent vector (compressed representation)
### Bottleneck (Latent Space)
- Low-dimensional embedding
- Captures most important features of the image
### Decoder
- Fully connected layers
- Expands latent vector back to 784 dimensions
- Output reshaped into 28×28 image

## Workflow
1. Load Fashion-MNIST dataset
2. Flatten images into vectors
3. Train encoder-decoder network
4. Minimize reconstruction loss (MSE / BCE)
5. Generate reconstructed images
6. Compare original vs reconstructed outputs

## Loss Function

- Mean Squared Error (MSE)
- Measures difference between original and reconstructed images

## Evaluation

- Visual comparison of reconstructed vs original images
- Reconstruction quality depends on:
  - Latent dimension size
  - Model depth
  - Training epochs

## Output

- Reconstructed images saved as:
  - `original(n).png`
  - `reconstructed(n).png`
- Results zipped for submission
