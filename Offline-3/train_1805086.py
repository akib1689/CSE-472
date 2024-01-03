from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from network import Network
from loss import categorical_cross_entropy, categorical_cross_entropy_prime
from activation import SigmoidActivation, ReLUActivation
from dense_layer import DenseLayer
from dropout_layer import DropoutLayer

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



# create the one hot encoder
one_hot_encoder = OneHotEncoder(categories='auto')

# fit the encoder to the labels
one_hot_encoder.fit(train_validation_labels)

# transform the labels using the one hot encoder
train_validation_labels = one_hot_encoder.transform(train_validation_labels).toarray()

# print the shape of the labels
print("Shape of labels: ", train_validation_labels.shape)



layers = [
    DenseLayer(784, 512),
    DropoutLayer(0.2),
    SigmoidActivation(),
    DenseLayer(512, 256),
    DropoutLayer(0.2),
    SigmoidActivation(),
    DenseLayer(256, 26)
]

net = Network(layers=layers,
                epochs=100,
                learning_rate=0.0005,
                decay_rate=0.99995,
                verbose=True,
                loss_function=categorical_cross_entropy,
                loss_function_prime=categorical_cross_entropy_prime,)


# Train and test split of the dataset


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



# train the network
net.train(X=train_data, Y=train_labels, val_X=validation_data, val_Y=validation_labels)

fig = plt.figure(figsize=(10, 15))

gs = gridspec.GridSpec(3, 2)  # create a 4x2 grid

# plot the loss curve
ax0 = plt.subplot(gs[0, :])
ax0.plot(range(net.epochs), net.losses)
ax0.set(xlabel="Epoch", ylabel="Loss")

# plot the accuracy curve
ax1 = plt.subplot(gs[1, 0])
ax1.plot(range(net.epochs), net.accuracy)
ax1.set(xlabel="Epoch", ylabel="Accuracy")

# plot the macro f1 score curve
ax2 = plt.subplot(gs[1, 1])
ax2.plot(range(net.epochs), net.f1_score)
ax2.set(xlabel="Epoch", ylabel="Macro F1 Score")

# plot the validation accuracy curve
ax3 = plt.subplot(gs[2, 0])
ax3.plot(range(net.epochs), net.val_accuracy)
ax3.set(xlabel="Epoch", ylabel="Validation Accuracy")

# plot the validation macro f1 score curve
ax4 = plt.subplot(gs[2, 1])
ax4.plot(range(net.epochs), net.val_f1_score)
ax4.set(xlabel="Epoch", ylabel="Validation Macro F1 Score")

plt.tight_layout()

# save the figure to a file
plt.savefig("training_curve_1805086.png")

# evaluate the network on the validation set
model_prediction = net.predict(validation_data)

validation_loss = np.mean(net.loss_function(validation_labels, model_prediction)) / len(validation_labels)

# calculate the accuracy of the model

from sklearn.metrics import accuracy_score, f1_score
# calculate the accuracy of the model
val_accuracy_score = accuracy_score(validation_labels.argmax(axis=1), model_prediction.argmax(axis=1))

# calculate the f1 score of the model
val_f1_score = f1_score(validation_labels.argmax(axis=1), model_prediction.argmax(axis=1), average='macro')



# print the loss of the model
print("Validation Loss of the model: ", validation_loss)

# print the accuracy of the model
print("Validation Accuracy of the model: ", val_accuracy_score)

# print the f1 score of the model
print("Validation macro-F1 score of the model: ", val_f1_score)



# predict training data
model_prediction = net.predict(train_data)

# calculate the loss on the training data
train_loss = np.mean(net.loss_function(train_labels, model_prediction)) / len(train_labels)

# calculate the accuracy of the model
train_accuracy_score = accuracy_score(train_labels.argmax(axis=1), model_prediction.argmax(axis=1))

# calculate the f1 score of the model
train_f1_score = f1_score(train_labels.argmax(axis=1), model_prediction.argmax(axis=1), average='macro')



# print the loss of the model
print("Training Loss of the model: ", train_loss)

# print the accuracy of the model
print("Training Accuracy of the model: ", train_accuracy_score)

# print the f1 score of the model
print("Training macro-F1 score of the model: ", train_f1_score)


import pickle

net = net.clear()

# save the model
with open("model.pkl", "wb") as file:
    pickle.dump(net, file)


# load the model
with open("model.pkl", "rb") as file:
    net = pickle.load(file)
    
# predict training data
model_prediction = net.predict(train_data)

# calculate the loss on the training data 
train_loss = np.mean(net.loss_function(train_labels, model_prediction)) / len(train_labels)

# calculate the accuracy of the model
train_accuracy_score = accuracy_score(train_labels.argmax(axis=1), model_prediction.argmax(axis=1))

# calculate the f1 score of the model
train_f1_score = f1_score(train_labels.argmax(axis=1), model_prediction.argmax(axis=1), average='macro')


# print the loss of the model
print("Training Loss of the model (after loading): ", train_loss)

# print the accuracy of the model
print("Training Accuracy of the model (after loding): ", train_accuracy_score)

# print the f1 score of the model
print("Training macro-F1 score of the model: (after loading)", train_f1_score)

