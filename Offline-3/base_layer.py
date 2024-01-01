# This is the base layer of the network, 
# Other layer classes will inherit from this class


class BaseLayer(object):
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        """Calculate the output of the layer

        Args:
            input (an array of inputs): The input to the layer used to calculate the output
        """
        pass
    
    def backward(self, output_gradient, learning_rate):
        """Calculate the delta of the layer

        Args:
            output_gradient (an array of gradients): The gradient of the output of the layer
            learning_rate (float): The learning rate of the network
        """
        pass