import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense_layer import DenseLayer
from activation import ReLUActivation, TanhActivation, SigmoidActivation
from loss import mean_squared_error, mean_squared_error_prime
from network import Network

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


# network = [
#     DenseLayer(2, 4),
#     TanhActivation(),
#     DenseLayer(4, 1),
#     TanhActivation()
# ]

# epochs = 100000
# learning_rate = 0.1

# losses = []

# # train the network
# for i in range(epochs):
#     loss = 0
#     for x, y in zip(X, Y):
#         # forward pass
#         output = x
#         for layer in network:
#             layer.forward(output)
#             output = layer.output

#         # calculate the loss
#         loss += mean_squared_error(y, output)

#         # backward pass
#         gradient = mean_squared_error_prime(y, output)
#         for layer in reversed(network):            
#             gradient = layer.backward(gradient, learning_rate)
            
#     loss /= len(X)
#     if i % 1000 == 0:
#         print("Epoch: %d, Loss: %.6f" % (i+1, loss))
#     losses.append(loss)
network = Network(number_of_inputs=2, number_of_outputs=1, 
                  number_of_layers=2, number_of_nodes_in_dense_layer=3,
                  activation_class=TanhActivation, epochs=100, batch_size=1)
network.train(X=X, Y=Y)


# print the output of the network for all the inputs
print("Output of the network for all the inputs:")
for x, y in zip(X, Y):
    output = network.predict(x)
    print("Input: %s, Output: %.6f" % (x, output))
    print(output)
    


# plot the loss
plt.plot(network.losses)
# plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

