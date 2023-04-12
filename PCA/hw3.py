from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    avg = np.mean(x, axis=0)
    centered = x - avg
    return centered

def get_covariance(dataset):
    # Your implementation goes here!
    trans = np.transpose(dataset)
    S = np.dot(trans, dataset)

    return S

def get_eig(S, m):
    # Your implementation goes here!
    eigenValues, eigenVectors = eigh(S, subset_by_index = [len(S)-m,len(S)-1])
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return np.diag(eigenValues), eigenVectors

def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigenValues, eigenVectors = eigh(S)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    variance_explained = eigenValues / np.sum(eigenValues)
    
    subset = variance_explained >= prop

    return np.diag(eigenValues[subset]), eigenVectors[:, subset]

def project_image(image, U):
    # Your implementation goes here!
    projected = np.dot(image, U)
    return np.dot(projected, U.T)

def display_image(orig, proj):
    # Your implementation goes here!
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (10,5))

    orig_im = np.reshape(orig, (32,32)).T
    ax1.set_title("Original")
    im1 = ax1.imshow(orig_im, aspect = "equal")
    fig.colorbar(im1, ax=ax1)

    proj_im = np.reshape(proj, (32,32)).T
    ax2.set_title("Projection")
    im2 = ax2.imshow(proj_im, aspect = "equal")
    fig.colorbar(im2, ax=ax2)

    return plt.show()