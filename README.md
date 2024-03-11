# Liver Tumors Semantic Segmentation From CT Scans Using Polar Transformations

## Overview
Focus of the project is on the impact of polar transformations on the performance of liver tumors semantic segmentation using Deep Convolutional Neural Networks, similar to [\[1\]][1].

## Data
The data used to train, validate the models is from the LiTS dataset and the test data is from the 3DIRCADB dataset.

## Models
For the purposes of this work, 3 popular semantic segmentation models have been considered: U-Net, U-Net++ with a ResNet encoder and DeepLabV3+ with  a ResNet encoder.

## Training
All models have been trained from scratch (no transfer learning has been done) using data augmentation techniques such as rotation, horizontal and vertical flip.
The training configuration is defined in the './src/model_training/training_config.py' file.
Training experiments have been handled, logged and monitored using the W&B service.

Each model has been trained in two separate settings: the carthesian setting and the polar setting, each of which are displayed in the diagrams below.

Carthesian Model Training Pipeline
![diagram1](https://github.com/vladtimug/MasterThesis/assets/44322734/283257c3-5427-4bf1-9b15-fd77d76a47a3)

Polar Model Training Pipeline
![diagram2](https://github.com/vladtimug/MasterThesis/assets/44322734/c5bd7525-1068-4e10-8dba-f9c3afbd0b94)

For the polar setting, the polar transformation was applied using as polar origin the center of mass of the biggest annotated blob in the corresponding mask for each CT scan slice.

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

U-Net Sample Prediction
![image](https://github.com/vladtimug/MasterThesis/assets/44322734/979c1ca6-8763-4ecc-925f-ddc026e98566)


U-Net++ Sample Prediction
![image](https://github.com/vladtimug/MasterThesis/assets/44322734/94e48990-c1cd-4213-91f9-b0c636b6a2ec)

DeepLabV3+ Sample Prediction
![image](https://github.com/vladtimug/MasterThesis/assets/44322734/2ab983ea-8974-498c-b5f1-67fb7697d8c4)


# References
* [1]: <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9551998> "Training on Polar Image Transformations Improves Biomedical Image Segmentation"
  [1]: "Training on Polar Image Transformations Improves Biomedical Image Segmentation"
