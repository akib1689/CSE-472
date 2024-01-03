## Load the model from the file
# and run test dataset
import torchvision.datasets as ds
import torchvision.transforms as transforms
import pickle
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

independent_test_dataset = ds.EMNIST(root='./data', 
                                split='letters',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)

# load the model from the file
# load the model
with open("model_1805086.pkl", "rb") as file:
    net = pickle.load(file)
    
    
    
test_data = []
test_labels = []
for data, label in independent_test_dataset:
    data_flatten = data.view(-1)
    test_data.append(data_flatten.numpy())
    test_labels.append(label)
    
test_data = np.array(test_data)
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1, 1)
    
    
    
# print the shape of the test dataset
print("Shape of the test dataset: ", test_data.shape)
print("Shape of the test labels: ", test_labels.shape)


one_hot_encoder = OneHotEncoder(categories='auto')

# fit the encoder to the labels
one_hot_encoder.fit(test_labels)

# transform the labels using the encoder
test_labels = one_hot_encoder.transform(test_labels).toarray()

# print the shape of the labels
print("Shape of labels(after converted): ", test_labels.shape)


# predict the output of the test dataset
prediction = net.predict(test_data)

# print the shape of the prediction
print("Shape of the prediction: ", prediction.shape)

# compare the prediction with the labels
acc = accuracy_score(test_labels.argmax(axis=1), prediction.argmax(axis=1))
f1 = f1_score(test_labels.argmax(axis=1), prediction.argmax(axis=1), average='macro')

# print the accuracy and f1 score
print("Accuracy: ", acc)
print("F1 score: ", f1)