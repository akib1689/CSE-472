# This is base activation class that inherits from the base layer class:

# other activation classes will inherit from this class

import numpy as np
from base_layer import BaseLayer

class BaseActivation(BaseLayer):
    def __init__(self, activation, activation_prime):
        """Initialize the activation function

        Args:
            activation_type (function): The type of the activation function
            activation_prime_type (function): The type of the derivative of the activation function
        """
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        """Calculate the output of the activation function

        Args:
            input (an array of inputs): The input to the activation function
        """
        self.input = input
        self.output = self.activation(self.input)
        
            
    def backward(self, output_gradient, learning_rate):
        """Calculate the delta of the activation function

        Args:
            output_gradient (an array of gradients): The gradient of the output of the activation function
            learning_rate (float): The learning rate of the network
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))