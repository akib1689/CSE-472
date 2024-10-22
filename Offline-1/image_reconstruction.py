"""
- Take the image (image.jpg) as an input
- Convert the image to grayscale using cv2.cvtColor from OpenCV (this will be a n*m matrix)
- Transform the matrix to lower dimension for faster computation (around 500 dimension)
- perform singular value decomposition on the matrix using numpy.linalg.svd
- Given a matrix A, and an intger k, write a function that returns the best rank k approximation of A
- Now vary the value of k from 1 to min(n, m) (take at least 10 such values in the interval). In each case, plot the resultant k-rank approximation as a grayscale image.
- Find the lowest k such that you can clearly read out the author's name from the image corresponding to the k-rank approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def fibonacci(n):
    """
    Given an integer n, return all the fibonacci numbers from 1 to n
    :param n: an integer
    :return: fibonacci numbers from 1 to n
    """
    numbers = [1, 1]
    for i in range(2, n):
        if numbers[i-1] + numbers[i-2] <= n:
            numbers.append(numbers[i-1] + numbers[i-2])
        else:
            break
    return numbers

def singular_value_decompose(A):
    """
    Given a matrix A, perform singular value decomposition on the matrix using numpy.linalg.svd
    :param A: A matrix
    :return: U, S, V
    """
    U, S, V = np.linalg.svd(A, full_matrices=False)
    return U, S, V

def best_rank_k_approximation(A, k):
    """
    Given a matrix A, and an intger k, returns the best rank k approximation of A
    :param A: A matrix
    :param k: an integer
    :return: A_k
    """
    U, S, V = singular_value_decompose(A)
    A_k = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    return A_k

def generate_k_rank_approximation(A, output_path):
    """
    From the given matrix A, generate k rank approximation by varying the value of k from 1 to min(n, m)
    :param A: A matrix
    :param output_path: output directory path
    :return: k rank approximation
    """
    intervals = fibonacci(min(A.shape))
    k_values = [1]
    for i in range(1, len(intervals)):
        if k_values[i-1] + intervals[i] <= min(A.shape):
            k_values.append(k_values[i-1] + intervals[i])
        else:
            break
    num_rows = 3
    num_cols = int(len(k_values) / num_rows) + 1
    print(num_rows , num_cols)
    for k in k_values:
        A_k = best_rank_k_approximation(A, k)
        plt.subplot(num_rows, num_cols, k_values.index(k)+ 1)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k = {k}')
        plt.axis('off')
        
    plt.savefig(os.path.join(output_path, 'k_rank_approximation.png'))

# Read the image
img = cv2.imread('image.jpg')
# calculate the dimensions of the image
dimensions = img.shape
print(dimensions)
# resize the image with aspect ratio keeping the same
width = 500
height = width * dimensions[0] / dimensions[1]
img = cv2.resize(img, (int(width), int(height)))
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to float64
gray = np.float64(gray)

print(gray.shape)

# perform singular value decomposition on the matrix using numpy.linalg.svd
U, S, V = singular_value_decompose(gray)

output_path = "output"


if not os.path.exists(output_path):
    os.makedirs(output_path)

generate_k_rank_approximation(gray, output_path)
