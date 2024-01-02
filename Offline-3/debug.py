import torchvision.datasets as ds
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from dense_layer import DenseLayer
from activation import SigmoidActivation
from loss import categorical_cross_entropy, categorical_cross_entropy_prime
import matplotlib.pyplot as plt

train_validation_set = ds.EMNIST(root='./data', 
                                split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

train_validation_data =[]
train_validation_labels = []

for data, label in train_validation_set:
    data_flatten = data.view(-1)
    train_validation_data.append(data_flatten.numpy())
    train_validation_labels.append(label)
    
train_validation_data = np.array(train_validation_data)
train_validation_labels = np.array(train_validation_labels)
# reshape the labels to be a column vector
train_validation_labels = train_validation_labels.reshape(-1, 1)

# create the one hot encoder
one_hot_encoder = OneHotEncoder(categories='auto')

# fit the encoder to the labels
one_hot_encoder.fit(train_validation_labels)

# transform the labels using the one hot encoder
train_validation_labels = one_hot_encoder.transform(train_validation_labels).toarray()


# split the training set into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(train_validation_data, train_validation_labels, 
    test_size=0.15, 
    random_state=42)

network = [
    DenseLayer(784, 128),
    SigmoidActivation(),
    DenseLayer(128, 26),
]

epochs = 10

batch_size = 64
learning_rate = 0.005

losses = []

def __forward__(layers, input):
    """This function is used to calculate the output of the network

    Args:
        input (array): The input to the network. 
                        This will be a single sample of the input data
    """
    output = input
    for layer in layers:
        layer.forward(output)
        output = layer.output
    return output

def __backward__(layers, output_gradient, learning_rate):
    """This function is used to calculate the delta of the network

    Args:
        output_gradient (array): The gradient of the output of the network
    """
    gradient = output_gradient
    for layer in reversed(layers):
        gradient = layer.backward(gradient, learning_rate)

# train the network
for epoch in range(epochs):
    loss = 0
    num_batches = len(train_data) // batch_size
    for batch in range(num_batches):
        x = train_data[batch * batch_size: (batch + 1) * batch_size]
        y = train_labels[batch * batch_size: (batch + 1) * batch_size]
        
        
        # forward pass
        output = __forward__(network, x)
        
        # calculate the loss
        loss += categorical_cross_entropy(y, output)
        # print("Loss: ", loss)
        loss = np.mean(loss)
        # backward pass
        gradient = categorical_cross_entropy_prime(y, output)
        
        __backward__(network, gradient, learning_rate)
            
    loss /= len(train_data)
    print("Current learning rate: ", learning_rate)
    print("Epoch: %d, Loss: %.8f" % (epoch+1, loss))
    losses.append(loss)
    
    
# plot the loss curve
# plt.plot(range(epochs), losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

# Test the network
def predict(network, X):
    """This function is used to predict the output of the network

    Args:
        X (array): The input to the network. 
                        This will be a single sample of the input data
    """
    output = __forward__(network, X)
    return output

def accuracy(network, X, y):
    """This function is used to calculate the accuracy of the network

    Args:
        X (array): The input to the network. 
                        This will be a single sample of the input data
        y (array): The labels for the data
    """
    predictions = predict(network, X)
    acc = accuracy_score(y.argmax(axis=1), predictions.argmax(axis=1))
    f1 = f1_score(y.argmax(axis=1), predictions.argmax(axis=1), average="weighted")
    return acc, f1

accuracy_score, f1_score = accuracy(network, validation_data, validation_labels)
print("Accuracy: ", accuracy_score)
print("F1 Score: ", f1_score)
