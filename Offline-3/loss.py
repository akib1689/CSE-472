# this is the loss function for the model

import numpy as np

# mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# derivative of mean squared error loss function
def mean_squared_error_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# binary cross entropy loss function
def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

# derivative of binary cross entropy loss function
def binary_cross_entropy_prime(y_true, y_pred):
    e = 1e-10
    y_pred = np.clip(y_pred, e, 1 - e)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# ! The following 2 functions are a bit rigid as it incorporates 
# ! the fact that the labels are one hot encoded.
# ! This can be improved by making the function more general
# categorical cross entropy loss function
def categorical_cross_entropy(y_true, y_pred):
    e = 1e-10
    y_pred = np.clip(y_pred, e, 1 - e)
    y_pred = softmax(y_pred)
    return -np.mean(y_true * np.log(y_pred), axis=1)

# derivative of categorical cross entropy loss function
def categorical_cross_entropy_prime(y_true, y_pred):
    e = 1e-10
    y_pred = np.clip(y_pred, e, 1 - e)
    y_pred = softmax(y_pred)
    return (y_pred - y_true)

def softmax(y):
    """Calculate the softmax of the vector y

    Args:
        y (an array of numbers): The input

    Returns:
        an array of numbers: The softmax of the input vector
    """
    # find the max value in the vector y
    max_value = np.max(y, axis=1, keepdims=True)
    y = y - max_value
    e = np.exp(y)
    return e / np.sum(e, axis=1, keepdims=True)