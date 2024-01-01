# Different types of activation functions

import numpy as np
from base_activation import BaseActivation

class TanhActivation(BaseActivation):
    def __init__(self):
        """Initialize the activation function
        """
        super().__init__(self.tanh, self.tanh_prime)
        
    def tanh(self, input):
        """Calculate the output of the activation function
        """
        return np.tanh(input)
    
    def tanh_prime(self, input):
        """Calculate the derivative of the activation function
        """
        return 1 - np.tanh(input) ** 2
    

class SigmoidActivation(BaseActivation):
    def __init__(self):
        """Initialize the activation function
        """
        super().__init__(self.sigmoid, self.sigmoid_prime)
        
    def sigmoid(self, input):
        """Calculate the output of the activation function
        """
        return 1 / (1 + np.exp(-input))
    
    def sigmoid_prime(self, input):
        """Calculate the derivative of the activation function
        """
        sigmoid = self.sigmoid(input)
        return sigmoid * (1 - sigmoid)
    
    
class ReLUActivation(BaseActivation):
    def __init__(self):
        """Initialize the activation function
        """
        super().__init__(self.relu, self.relu_prime)
        
    def relu(self, input):
        """Calculate the output of the activation function
        """
        return np.maximum(0, input)
    
    def relu_prime(self, input):
        """Calculate the derivative of the activation function
        """
        return (input > 0).astype(np.float32)