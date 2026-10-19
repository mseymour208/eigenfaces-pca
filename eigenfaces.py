import scipy as sp
import numpy as np
from PIL import Image
from pathlib import Path

# Considering a set of m images
# Convert each image into 1D vectors
# Find the average of all the face vectors:
#   y = 1/m * the sum from i = 1 to m of x_i
# Subtract the mean from each vector
#   a_i = x_i - y
# Create a matrix of all the new vectors, A
# Find the covariance matrix (how all the vectors change together)
#   A * A_transpose -> N^2 x N^2 is not computationally efficient
#   A_transpose * A -> M x M (assuming M < N^2)
#   Cov = A_transpose * A
# Find the eigenvectors of the covariance matrix
#   C' * u_i = lambda_i * u_i
#   Where C' = A * A_transpose and u_i = A * v_i
# Select the k largest eigenvectors of C'where k < M
# We represent each normalized face vector as a linear combination of the best k eigenvectors
#   x_i - y = the sum from j = 1 to k of w_j * u_j, where w_j is the coefficient and u_j are our eigenfaces
# Represent the training faces in the form of a vector of the coefficients


# Loading our training images

training_paths = [] # List of paths to each training image
image_vectors = [] # List of training image vectors
training_folder = Path() # Path to folder of training images

# Populate image_vectors with vectors for each training image
for img_path in training_folder.glob('*.jpg'):
    with Image.open(img_path) as img:
        img = img.convert('L') # Convert to greyscale
        img = img.resize((200, 200))

        img_vec = np.array(Image.open(img_path)).flatten().astype(np.float64) # flattening to 1D
        image_vectors.append(img_vec)
        training_paths.append(img_path)
