"""_summary_
- Take the dimentions of matrix n as an input
- Generate a random invertible symmetric matrix of size n x n. Each value in the matrix is a random integer.
- Find the eigenvalues and eigenvectors of the matrix. using numpy.linalg.eig
- Reconstruct the matrix using the eigenvalues and eigenvectors. using numpy.dot
- Compare the original matrix with the reconstructed matrix. using numpy.allclose 
"""

import numpy as np

def random_invertible_symmetric_matrix(n):
    """_summary_
    Generate a random invertible symmetric matrix of size n x n. Each value in the matrix is a random integer.
    """
    # Generate a random matrix of size n x n
    A = np.random.randint(10, size=(n, n))
    # Check if the matrix is invertible
    while np.linalg.det(A) == 0:
        # Generate a random matrix of size n x n
        A = np.random.randint(10, size=(n, n))
    # Make the matrix symmetric
    A = (A + A.T) // 2
    return A


def compare_matrices(A, B):
    """_summary_
    Compare the original matrix with the reconstructed matrix. using numpy.allclose
    """
    return np.allclose(A, B)

# Take the dimentions of matrix n as an input
n = int(input("Enter the dimentions of matrix n: "))
# Generate a random invertible symmetric matrix of size n x n. Each value in the matrix is a random integer.
A = random_invertible_symmetric_matrix(n)

# p[rint the original matrix

print("Original matrix: \n", A)


# Find the eigenvalues and eigenvectors of the matrix. using numpy.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues: \n", eigenvalues)
print("Eigenvectors: \n", eigenvectors)

# Reconstruct the matrix using the eigenvalues and eigenvectors. using numpy.dot
A_reconstructed = np.dot(np.dot(eigenvectors,np.diag(eigenvalues)),np.linalg.inv(eigenvectors))

# Compare the original matrix with the reconstructed matrix. using numpy.allclose
print("Are the original matrix and the reconstructed matrix the same? ", compare_matrices(A, A_reconstructed))