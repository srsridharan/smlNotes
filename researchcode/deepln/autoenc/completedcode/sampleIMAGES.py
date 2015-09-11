from scipy.io import loadmat as loader
import numpy as np
import ipdb as pdb
def sampleIMAGES():
    """
    returns  @patches
    """
    # sampleIMAGES
    # Returns 10000 patches for training

    IMAGES = loader('./IMAGES.mat')['IMAGES']


    patchsize = 8;  # we'll use 8x8 patches
    numpatches = 10000;

    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns.
    patches = np.zeros((patchsize*patchsize, numpatches))

    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Fill in the variable called "patches" using data
    #  from IMAGES.
    #
    #  IMAGES is a 3D array containing 10 images
    #  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
    #  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
    #  it. (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
    #  patch corresponding to the pixels in the block (21,21) to (30,30) of
    #  Image 1
    imagechoice = np.random.randint(0, np.shape(IMAGES)[2], (numpatches,1))
    indxes = np.random.randint(0, np.shape(IMAGES)[0]-8, (numpatches, 2))
    for k in range(numpatches):
        littlepatch = IMAGES[indxes[k,0]:indxes[k,0]+8, indxes[k,1]:indxes[k,1]+8,imagechoice[k]]
        patches[:, k] = littlepatch.flatten()

    ## ---------------------------------------------------------------
    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure
    # the range of pixel values is also bounded between [0,1]
    patches = normalizeData(patches)

    return patches

## ---------------------------------------------------------------
def  normalizeData(patches):
    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer

    # Remove DC (mean of images).
    patches = patches - np.mean(patches,0)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd;

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1;

    return patches
