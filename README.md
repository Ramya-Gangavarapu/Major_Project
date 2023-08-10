# Major_Project

# Implementation of Neural Networks for Color Classification and and Real Time Color Prediction for Image Processing

# TEAM MEMBERS
G. Ramya 

G. Sanjana 

B. Prabhakar 

K. Sai Ram 

## OBJECTIVES

Implementing a custom Convolutional Neural Network (CNN) from scratch in MATLAB for color classification.
Utilizing the TCS3200 color sensor to detect and capture color information.
Developing a CNN-based model to accurately classify colors under ideal conditions.
Extending the CNN model to handle non-ideal conditions and variations in color detection.
Integrating a webcam to capture live images for real-time color prediction.
Transforming the MATLAB code into C code using Energia software for compatibility with the MSP430F5529LP microcontroller.
Establishing hardware connections between the MSP430F5529LP microcontroller, TCS3200 color sensor, and a 16x2 LCD display.
Displaying the predicted color and RGB values on the LCD display in real-time.
Evaluating and optimizing the performance of the CNN model for accurate color classification.
Demonstrating the effectiveness and practical applicability of the implemented CNN solution for color analysis.

## INTRODUCTION

This project focuses on implementing a custom CNN for real-time color classification and prediction using a webcam. Traditional color classification methods may struggle with complex color distributions, so deep learning techniques, particularly CNNs, are employed. The TCS3200 color sensor detects colors, and a webcam captures live images. The implemented CNN, developed in MATLAB, extracts and assigns significance to image features for accurate color classification. The Keras library is used for implementing a multi-input CNN model. The Energia software platform enables practical implementation on the MSP430F5529LP microcontroller, with hardware components like the LCD display for real-time color analysis. The project showcases the effectiveness of the custom CNN approach for accurate color classification in various conditions.

## CNN LAYERS

Convolutional Layer: This layer performs the convolution operation by applying a set of filters or kernels to the input image or feature maps. It extracts features and reduces the spatial dimensions of the input.
Pooling Layer: This layer performs pooling operations such as max pooling or average pooling to down sample the output of the convolutional layer. It reduces the spatial dimensions of the input while preserving the important features.
Fully Connected Layer: This layer connects all neurons in the previous layer to all neurons in the next layer, and performs a linear transformation followed by a non-linear activation function.
Dropout Layer: This layer randomly drops out a proportion of the inputs to the next layer during training to prevent overfitting.
Batch Normalization Layer: This layer normalizes the input to each layer, improving the stability of the network during training.
Activation Layer: This layer applies a non-linear activation function to the output of the previous layer.
Softmax Layer: This layer is typically used as the output layer for classification problems. It computes the probabilities of each class and outputs the class with the highest probability.

## FEED FORWARD NEURAL NETWORK 

A feedforward neural network (FFNN) is a type of artificial neural network that consists of an input layer, one or more hidden layers, and an output layer. The layers are connected in a feedforward manner, meaning that the output of one layer serves as the input for the next layer.
The input layer is responsible for receiving the input data, which is then passed through the hidden layers to the output layer. The output layer produces the final output of the network.
Each neuron in the hidden layers and the output layer is associated with a weight, which determines the strength of its connection to other neurons in the network. These weights are adjusted during the training process to improve the performance of the network.
FFNNs are commonly used for classification and regression tasks. In a classification task, the output of the network represents the probability of a given input belonging to each class. In a regression task, the output represents a continuous value.
FFNNs use an activation function to introduce non-linearity into the network. Common activation functions include sigmoid, tanh, and ReLU.
The training process of an FFNN involves forward propagation, which involves passing the input data through the network and calculating the output, and backward propagation, which involves adjusting the weights of the neurons in the network based on the error between the predicted output and the actual output.
FFNNs have been successful in a wide range of applications, including image recognition, speech recognition, and natural language processing.

## COLOR CLASSIFICATION USING FFNS

Feed forward neural networks (MLPs) are used for color classification and detection tasks.
MLPs consist of layers of neurons with non-linear activation functions.
Preprocessing is done to extract color information from input data.
Input data is converted to color space and color channels are extracted.
Each neuron in the first layer represents a specific color channel.
The network is trained using a labeled dataset.
Weights and biases of neurons are adjusted through optimization during training.
Trained network can classify new color samples.
Input data is fed into the network, and the output represents the predicted color class.
Predicted color class is compared to the actual color class for evaluation.

## ADVANTAGES OF CNN - KNN

Neural networks can learn complex patterns and relationships in data, making them suitable for color classification tasks.
They can handle non-linear relationships between color features, allowing for accurate classification.
Neural networks can adapt and generalize well to different color distributions and lighting conditions.
They can be trained on large datasets, improving their performance and robustness.
Neural networks can process images in real-time, enabling live color prediction using a webcam.

## APPLICATIONS 

Object recognition: Neural networks can be used to classify colors of objects in images or videos, enabling object recognition and tracking systems.

Quality control: Neural networks can assist in color-based quality control processes, such as identifying defective or mismatched products based on color characteristics.

Sorting and categorization: Neural networks can be utilized in color-based sorting systems, where objects or materials are classified and sorted based on their color attributes.

Image processing: Neural networks can enhance color-based image processing tasks, such as color correction, image segmentation, and image retrieval.

Industrial automation: Neural networks can be integrated into industrial automation systems to classify and analyze colors in real-time, facilitating automated decision-making processes.

Robotics: Neural networks can support color perception in robotic systems, enabling robots to interact with and respond to color-coded objects or environments.

Biomedical imaging: Neural networks can aid in color-based analysis and interpretation of medical images, assisting in areas such as tissue segmentation and disease diagnosis.

Gaming and augmented reality: Neural networks can be used to recognize and track colors in gaming and augmented reality applications, enabling interactive and immersive experiences.

Traffic monitoring: Neural networks can analyze color information from traffic surveillance cameras to detect and track vehicles, monitor traffic flow, and identify violations.

## REFERENCES

[1]. Wang, Y., Ma, Z., Li, G., & Yuan, Y. (2015). Color image classification using local color features and SVM. Signal Processing, 111, 49-59.

[2]. "A CNN-Based Real-Time Facial Expression Recognition System on Raspberry Pi" by S. Kumari and V. Tyagi in the International Journal of Advanced Trends in Computer Science and Engineering, Volume 8 - Issue 3, May-June 2019.

[3]. "Liu, T., Liu, Y., Wu, F., & Chen, J. (2018). An efficient color-based image retrieval system using color and texture features. Multimedia Tools and Applications, 77(6), 6947-6966.

[4]. "Basso, L., Diaz, M., & Veloso, M. (2016). Learning color categories for object detection. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 4057-4062).

[5]. "Jin, X., Ma, Z., Huang, M., & Chen, S. (2017). Review on color image segmentation methods. Pattern Recognition, 63, 155-171.

[6]. "Sun, H., Jin, L., Liu, Z., & Lin, Y. (2015). Color image enhancement based on local adaptive color correction. Signal Processing, 107, 430-441.


