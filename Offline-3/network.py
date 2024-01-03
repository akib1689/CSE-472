# This file contains the network class, which is used to create a network of nodes

from activation import ReLUActivation
from loss import mean_squared_error, mean_squared_error_prime
from dense_layer import DenseLayer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class Network:
    def __init__(self, number_of_layers = None, number_of_inputs = None, number_of_outputs = None, number_of_nodes_in_dense_layer = None,  activation_class = ReLUActivation, learning_rate = 0.2, epochs = 100, verbose = False, loss_function = mean_squared_error, loss_function_prime = mean_squared_error_prime, batch_size = 64, decay_rate = None, layers = None):
        """constructor for the network class. This function is used to create a feed forward neural network

        Args:
            number_of_layers (int): The number of layers in the network
            number_of_inputs (int): The number of inputs to the network
            number_of_outputs (int): The number of outputs from the network
            number_of_nodes_in_dense_layer (int): The number of nodes in the dense layer
            activation_class (class): The class of the activation function to be used
            learning_rate (float): The learning rate of the network
            epochs (int): The number of epochs to train the network
            verbose (bool): Whether to print the loss after each epoch
            loss_function (function): The loss function to be used
            loss_function_prime (function): The derivative of the loss function to be used
            batch_size (int): The batch size to be used for training
        """
        # check if the number of layers, number of inputs, number of outputs, and number of nodes in dense layer are given or Layers are given
        if not number_of_layers and not number_of_inputs and not number_of_outputs and not number_of_nodes_in_dense_layer and not layers:
            raise Exception("Please provide either number of layers, number of inputs, number of outputs, and number of nodes in dense layer or layers")
        if number_of_inputs and number_of_outputs and number_of_nodes_in_dense_layer and number_of_layers:
            self.number_of_layers = number_of_layers * 2
            self.number_of_inputs = number_of_inputs
            self.number_of_outputs = number_of_outputs
            self.number_of_nodes_in_dense_layer = number_of_nodes_in_dense_layer
        self.activation_class = activation_class
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime
        if decay_rate:
            self.decay_rate = decay_rate
        else:
            self.decay_rate = 1- 1/(epochs+1)
        if layers:
            self.layers = layers
        else:
            self.layers = []
            self.__create_layers__()
        self.losses = []
        self.accuracy = []
        self.f1_score = []
        if self.verbose:
            print("Network created with the following layers:")
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    print("Layer %s: input size %s and output size %s" % (layer.__class__.__name__, layer.input_size, layer.output_size))
                else:
                    print("Layer %s" % (layer.__class__.__name__))
        
    
        
        
    def __create_layers__(self):
        """This function is used to create the layers of the network
        """
        for layer in range(self.number_of_layers):
            if layer == 0:
                # This will be the input layer
                self.layers.append(DenseLayer(self.number_of_inputs, self.number_of_nodes_in_dense_layer))
            elif layer == self.number_of_layers - 2:
                # This will be the output layer
                self.layers.append(DenseLayer(self.number_of_nodes_in_dense_layer, self.number_of_outputs))
            elif layer == self.number_of_layers - 1:
                # don't add an activation layer after the output layer
                continue                             
            elif layer % 2 == 0:
                # This will be a fully connected layer (dense layer)
                self.layers.append(DenseLayer(self.number_of_nodes_in_dense_layer, self.number_of_nodes_in_dense_layer))
            else:
                # This will be an activation layer
                self.layers.append(self.activation_class())
    
    def predict(self, input):
        """This function is used to predict the output of the network

        Args:
            input (array): The input to the network. 
                           This will be a single sample of the input data

        Returns:
            array: The output of the network
        """
        return self.__forward__(input)
    
    def __forward__(self, input):
        """This function is used to calculate the output of the network

        Args:
            input (array): The input to the network. 
                           This will be a single sample of the input data
        """
        output = input
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def __backward__(self, output_gradient):
        """This function is used to calculate the delta of the network

        Args:
            output_gradient (array): The gradient of the output of the network
        """
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, self.learning_rate)
            
    
    def train(self, X, Y):
        """This function is used to train the network

        Args:
            X (array): The input to the network
            Y (array): The output of the network
        """
        print("Training the network...")
        for epoch in range(self.epochs):
            loss = 0
            num_batches = len(X) // self.batch_size
            for batch in range(num_batches):
                x = X[batch * self.batch_size: (batch + 1) * self.batch_size]
                y = Y[batch * self.batch_size: (batch + 1) * self.batch_size]
                
                
                # forward pass
                output = self.__forward__(x)
                
                # calculate the loss
                loss += self.loss_function(y, output)
                # print("Loss: ", loss)
                loss = np.mean(loss)
                # backward pass
                gradient = self.loss_function_prime(y, output)
                
                self.__backward__(gradient)
                    
            loss /= len(X)
            prediction = self.predict(X)
            accuracy = accuracy_score(Y.argmax(axis=1), prediction.argmax(axis=1))
            f1 = f1_score(Y.argmax(axis=1), prediction.argmax(axis=1), average='macro')
            self.losses.append(loss)
            self.accuracy.append(accuracy)
            self.f1_score.append(f1)
            self.learning_rate *= self.decay_rate
            if self.learning_rate < 0.0001:
                self.learning_rate = 0.0001
            if self.verbose:
                print("Epoch: %d" % (epoch+1))
                print("\tLoss: %.8f" % (loss))
                print("\tAccuracy: %.8f" % (accuracy))
                print("\tF1 Score: %.8f" % (f1))
                print("\tCurrent learning rate: ", self.learning_rate)
    
        
                