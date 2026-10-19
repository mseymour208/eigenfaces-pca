import scipy as sp
import numpy as np
from PIL import Image
from pathlib import Path
from functools import reduce


# Considering a set of m images
# Convert each image into 1D vectors

training_paths = [] # List of paths to each training image
image_vectors = [] # List of training image vectors
training_folder = Path('C:\\Users\\Mikle\\OneDrive\\Desktop\\eigenfaces-pca\\TrainingImages') # Path to folder of training images

# Populate image_vectors with vectors for each training image
for img_path in training_folder.glob("*"):
    with Image.open(img_path) as img:
        img = img.convert('L').resize((200, 200)) # Convert to greyscale

        img_vec = np.array(Image.open(img_path)).flatten().astype(np.float64) # flattening to 1D
        image_vectors.append(img_vec)
        training_paths.append(img_path)


# Find the average of all the face vectors:
#   y = 1/m * the sum from i = 1 to m of x_i
# Subtract the mean from each vector
#   a_i = x_i - y
# Create a matrix of all the new vectors, A

# Finding the average vector

# Converting list of 1D image vectors to 2D matrix of m images x n pixels
image_matrix = np.array(image_vectors)
vec_average = np.mean(image_matrix, axis=0)

A = image_matrix - vec_average
covariance = np.dot(A, A.T)

# Finding the eigenvalues & eigenvectors of the covariance matrix
e_vals, e_vecs = np.linalg.eigh(covariance)
idx = np.argsort(e_vals)[::-1]

# Creating a matrix of eigenvalues in descending order
e_vals = e_vals[idx]
e_vecs = e_vecs[idx]

# Creating the eigenfaces => matrix of the transpose of centered faces * their eigenvectors
eigenfaces = np.dot(A.T, e_vecs)


