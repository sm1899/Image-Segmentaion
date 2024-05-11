# Image Segmentation using MobileNetEncoderDecoder

This project focuses on image segmentation using the MobileNetEncoderDecoder model on the ISIC 2016 dataset. The model is trained to segment skin lesions in dermoscopic images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to develop an image segmentation model that can accurately segment skin lesions in dermoscopic images. The model utilizes a MobileNet-based encoder-decoder architecture to achieve efficient and accurate segmentation.

## Dataset

The ISIC 2016 dataset is used for training and evaluation. It consists of dermoscopic images along with their corresponding ground truth segmentation masks. The dataset provides a diverse collection of skin lesion images, enabling the model to learn and generalize well.

## Model Architecture

The model architecture consists of two main components:

1. **MobileNetEncoder**: The encoder is based on the MobileNetV2 architecture, which is pretrained on the ImageNet dataset. It extracts features from the input image at different scales using skip connections.

2. **CustomDecoder**: The decoder takes the output of the encoder and upsamples it using transposed convolutions and skip connections from the encoder. It gradually increases the spatial resolution of the feature maps and produces the final segmentation mask.

The encoder can be optionally frozen during training to speed up the training process and prevent overfitting.

## Loss Function

The loss function used for training the model is a weighted sum of three components:

1. **Binary Cross-Entropy (BCE) Loss**: Measures the pixel-wise binary cross-entropy between the predicted segmentation mask and the ground truth mask.

2. **Dice Loss**: Computes the Dice coefficient between the predicted and ground truth masks, encouraging spatial overlap and reducing false positives and false negatives.

3. **Intersection over Union (IoU) Loss**: Calculates the intersection over union between the predicted and ground truth masks, promoting accurate segmentation boundaries.

The weights of each loss component can be adjusted based on the desired emphasis on each metric.

## Installation

To run this project locally, please follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/image-segmentation-mobilenet.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

## Results

The model achieves promising results on the ISIC 2016 dataset. The segmentation performance is evaluated using metrics such as Dice coefficient, IoU, and pixel-wise accuracy. Detailed results and visualizations can be found in the `Jupyter Notebook`  

## License

This project is licensed under the [MIT License](LICENSE).

