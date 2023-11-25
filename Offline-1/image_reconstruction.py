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
import cv2
import matplotlib.pyplot as plt
import os

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

def generate_k_rank_approximation(A, k, output_path):
    """
    From the given matrix A, generate k rank approximation by varying the value of k from 1 to min(n, m) at least k such values in the interval
    :param A: A matrix
    :param k: an integer (number of intervals)
    :param output_path: output directory path
    :return: k rank approximation
    """
    interval = int(min(A.shape[0], A.shape[1]) / k)
    k_values = [i for i in range(1, min(A.shape[0], A.shape[1]), interval)]
    num_rows = 3
    num_cols = int(len(k_values) / num_rows) + 1
    print(num_rows , num_cols)
    print(len(k_values))
    for k in k_values:
        A_k = best_rank_k_approximation(A, k)
        plt.subplot(num_rows, num_cols, k_values.index(k)+ 1)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k = {k}')
        plt.axis('off')
    plt.savefig(f'{output_path}/k_rank_approximation.png')

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

# print(f'Left orthogonal matrix C:\n{np.round(U, 2)}\n')
# print(f'Singular values diagonal matrix C:\n{np.round(S, 2)}\n')
# print(f'Right orthogonal matrix C:\n{np.round(V, 2)}')
output_path = "output"
# if the direntory does not exist, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)
generate_k_rank_approximation(gray, 17, output_path)





