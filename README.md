# MLP_to_classify_MNIST
This project aims at using numpy to build a multilayer perceptron neural network. Dataset is MNIST image. Splitting 60000 images into 50000 training images and 10000 validation images.
The train h5 file is too large to upload.

It has 784 neurons in the input layer and 10 neurons in the output layer.  
Hidden layer activations: I try tanh and ReLU and regard this as hyperparameter which is chosen by validation.  
Cost function: Cross entropy.  
Optimizer: Stochastic gradient descent with learning rate in [0.02, 0.1, 0.5], also as hyperparameter.  
Minibatch size: 500.  
Train the network for 50 epoches. At epoch 20 and 40, learning rate decay by a half.

For outcome, please see report.
