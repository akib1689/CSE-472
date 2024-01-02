#!/usr/bin/env python
# coding: utf-8

# # Offline on the Feed Forward Neural Network
# 
# ### Introduction
# 
# Each layers implementation can be found in the `layers.py` file. The `network.py` file contains the implementation of the network. The forward pass and backward pass is implemented here. The same file contains the training code also predict function which is used to predict the output of the network.

# In[ ]:


import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np

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


# In[ ]:


# print the number of samples in the training set
print("Number of samples in the training set: ", len(train_validation_data))

# print the shape of each sample
print("Shape of each sample: ", train_validation_data.shape)

# print the shape of the labels
print("Shape of labels: ", train_validation_labels.shape)

# reshape the labels to be a column vector
train_validation_labels = train_validation_labels.reshape(-1, 1)

# print the shape of the labels
print("Shape of labels: ", train_validation_labels.shape)

# print the number of different labels in the training set
print("Number of different labels: ", len(np.unique(train_validation_labels)))


# ### Plot the first 5 images in the training dataset

# In[ ]:


import matplotlib.pyplot as plt

# plot the first 5 samples in the training set
fig = plt.figure(figsize=(20, 5))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1)
    ax.imshow(train_validation_data[i].reshape(28, 28), cmap='gray')
    ax.set_title("Label: {}".format(train_validation_labels[i]))
plt.show()


# ### One hot encoding of labels
#  
# We want to convert the labels into one hot encoding. The one hot encoding is a vector of length equal to the number of classes. 
# The vector is all zeros except for the class which is represented by a one. For example, 
# - if the class is 3 and the total number of classes is 5 then the one hot encoding will be `[0, 0, 0, 1, 0]`. The one hot encoding of the labels is stored in the variable `y_oh`.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

print("Shape of labels: ", train_validation_labels.shape)


# create the one hot encoder
one_hot_encoder = OneHotEncoder(categories='auto')

# fit the encoder to the labels
one_hot_encoder.fit(train_validation_labels)

# transform the labels using the one hot encoder
train_validation_labels = one_hot_encoder.transform(train_validation_labels).toarray()

# print the shape of the labels
print("Shape of labels: ", train_validation_labels.shape)



# 
# ### Create the network with the following architecture
# 
# - Input layer with 784 neurons
# - Hidden layer with 256 neurons
# - Output layer with 26 neurons
# - Activation function for hidden layer: ReLU
# - Number of hidden layers: 2
# - Number of epochs: 10

# In[ ]:


from network import Network
from activation import SigmoidActivation
from loss import categorical_cross_entropy, categorical_cross_entropy_prime


# create a network object
net = Network(number_of_inputs=784,
              number_of_outputs=26,
              number_of_layers=3,
              number_of_nodes_in_dense_layer=256,
              epochs=20,
              verbose=True,
              learning_rate=0.0005,
              decay_rate=1,
              activation_class=SigmoidActivation,
              loss_function=categorical_cross_entropy,
              loss_function_prime=categorical_cross_entropy_prime,)


# ### Train and test split of the dataset
# 
# The dataset is split into training and testing dataset. The training dataset is used to train the network and the testing dataset is used to test the network. The training dataset is 80% of the total dataset and the testing dataset is 20% of the total dataset. The training dataset is stored in the variables `X_train` and `y_train`. The testing dataset is stored in the variables `X_test` and `y_test`.

# In[ ]:


from sklearn.model_selection import train_test_split

# split the training set into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(train_validation_data, 
                                                                                train_validation_labels, 
                                                                                test_size=0.15, 
                                                                                random_state=42)

# check the shape of the training data
print("Shape of training data: ", train_data.shape)
print("Shape of training labels: ", train_labels.shape)

# check the shape of the validation data
print("Shape of validation data: ", validation_data.shape)
print("Shape of validation labels: ", validation_labels.shape)


# ### Train the network
# 
# We can train the network by calling the `train` function. The `train` function takes the following parameters:
# 
# - `X`: Training data
# - `Y`: Training labels

# In[ ]:


# train the network
net.train(train_data, train_labels)


# ### Plot the loss curve

# In[ ]:


# plot the loss curve
plt.plot(range(net.epochs), net.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# ### Predict the output of the network
# 
# We can predict the output of the network by calling the `predict` function. The `predict` function takes the following parameters:

# In[ ]:


# evaluate the network on the validation set

model_prediction = net.predict(validation_data)


# calculate the accuracy of the model

from sklearn.metrics import accuracy_score, f1_score
# calculate the accuracy of the model
accuracy_score = accuracy_score(validation_labels.argmax(axis=1), model_prediction.argmax(axis=1))

# calculate the f1 score of the model
f1_score = f1_score(validation_labels.argmax(axis=1), model_prediction.argmax(axis=1), average='weighted')


# print the accuracy of the model
print("Accuracy of the model: ", accuracy_score)

# print the f1 score of the model
print("F1 score of the model: ", f1_score)



