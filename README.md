# Semantic Segmentation of Liver Tumors using Polar Transformations

## Overview
Focus of the project is on the impact of polar transformations on the performance of liver tumors semantic segmentation using Deep Convolutional Neural Networks.

## Data
The data used to train, validate the models is from the LiTS dataset and the test data is from the 3DIRCADB dataset.

## Models
For the purposes of this work, 3 popular semantic segmentation models have been considered: U-Net, U-Net++ with a ResNet encoder and DeepLabV3+ with  a ResNet encoder.

## Training
All models have been trained from scratch (no transfer learning has been done) using data augmentation techniques such as rotation, horizontal and vertical flip.
The training configuration is defined in the './src/model_training/training_config.py' file.
Training experiments have been handled, logged and monitored using the W&B service.

## Results
### Carthesian Models
| Model | Average Test Dice Score |
| :---:   | :---: |
| U-Net | 78.67%   |
| U-Net++ | 75.48% |
| DeepLabV3+ | 80.92% |

### Polar Models
| Model | Average Test Dice Score |
| :---:   | :---: |
| U-Net | 64.12%   |
| U-Net++ | 67.26% |
| DeepLabV3+ | 52.22% |
