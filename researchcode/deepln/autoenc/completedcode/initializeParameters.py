import numpy as np
from numpy import sqrt


def initializeParameters(hiddenSize, visibleSize):

    """
    returns @theta
    """
    # Initialize parameters randomly based on layer sizes.
    r = sqrt(6)/sqrt(hiddenSize + visibleSize + 1)
    # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.random((hiddenSize, visibleSize)) * 2 * r - r
    W2 = np.random.random((visibleSize, hiddenSize)) * 2 * r - r
    b1 = np.zeros((hiddenSize, 1))
    b2 = np.zeros((visibleSize, 1))
    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    theta = np.hstack([W1.flatten(), W2.flatten(),  b1.flatten(), b2.flatten()])
    return theta
