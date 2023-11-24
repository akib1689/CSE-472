"""
- Take the image (image.jpg) as an input
- Convert the image to grayscale using cv2.cvtColor from OpenCV (this will be a n*m matrix)
- Transform the matrix to lower dimension for faster computation
- perform singular value decomposition on the matrix using numpy.linalg.svd
- Given a matrix A, and an intger k, write a function that returns the best rank k approximation of A
- Now vary the value of k from 1 to min(n, m) (take at least 10 such values in the interval). In each case, plot the resultant k-rank approximation as a grayscale image.
- Find the lowest k such that you can clearly read out the author's name from the image corresponding to the k-rank approximation.
"""

import numpy as np
