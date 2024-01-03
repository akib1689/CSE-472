# This is the dropout layer:
# inherit from the base layer class

import numpy as np

from base_layer import BaseLayer

class DropoutLayer(BaseLayer):
    def __init__(self, dropout_rate):
        """This function is used to initialize the dropout layer
        
        Args:
            dropout_rate (float): The probability of dropping a neuron
        """
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        
    def forward(self, input):
        """This function is used to perform the forward pass
        
        Args:
            inputs (numpy array): The input to the layer
            
        Returns:
            numpy array: The output of the layer
        """
        self.input = input
        self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape)
        self.output = np.multiply(self.input, self.dropout_mask)
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """This function is used to perform the backward pass
        
        Args:
            output_gradient (numpy array): The gradient of the output of the layer
            learning_rate (float): The learning rate of the neural network
            
        Returns:
            numpy array: The gradient of the input of the layer
        """
        return np.multiply(output_gradient, self.dropout_mask)
    
    def clear(self):
        """Clear the input and output of the layer
        """
        super().clear()
        self.dropout_mask = None