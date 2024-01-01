# This is one of the layer classes that inherits from the base layer class:

import numpy as np
from base_layer import BaseLayer

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        """Initialize the weights and biases of the layer

        Args:
            input_size (int): The size of the input
            output_size (int): The size of the output
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) + 1
        self.biases = np.random.randn(output_size, 1) 
        
    def forward(self, input):
        """Calculate the output of the layer

        Args:
            input (an array of inputs): The input to the layer used to calculate the output
        """
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        
    def backward(self, output_gradient, learning_rate):
        """Calculate the delta of the layer

        Args:
            output_gradient (an array of gradients): The gradient of the output of the layer
            learning_rate (float): The learning rate of the network
        """
        input_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        biases_gradient = output_gradient.mean(axis=0) * self.input.shape[0]
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_gradient