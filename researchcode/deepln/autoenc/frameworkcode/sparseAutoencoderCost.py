import numpy as np

def  sparseAutoencoderCost(theta, visibleSize, hiddenSize,
                                             lam, sparsityParam, beta, data):
    """
    returns @cost, @grad

    """

    # visibleSize: the number of input units (probably 64)
    # hiddenSize: the number of hidden units (probably 25)
    # lam: weight decay parameter
    # sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    #                           notes by the greek alphabet rho, which looks like a lower-case "p").
    # beta: weight of sparsity penalty term
    # data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = np.reshape(theta[0:hiddenSize*visibleSize], (hiddenSize, visibleSize))
    W2 = np.reshape(theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize], (visibleSize, hiddenSize))
    b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize:]

    # Cost and gradient variables (your code needs to compute these values).
    # Here, we initialize them to zeros.
    cost = 0
    W1grad = np.zeros(np.shape(W1))
    W2grad = np.zeros(np.shape(W2))
    b1grad = np.zeros(np.shape(b1))
    b2grad = np.zeros(np.shape(b2))

    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    #
    # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
    # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
    # [(1/m) \Delta W^{(1)} + \lam W^{(1)}] in the last block of pseudo-code in Section 2.2
    # of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    #
    # Stated differently, if we were using batch gradient descent to optimize the parameters,
    # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
    #


















    #-------------------------------------------------------------------
    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.

    grad = np.hstack([W1grad.flatten(),  W2grad.flatten(),  b1grad.flatten(),  b2grad.flatten()])

    return [cost, grad]

#-------------------------------------------------------------------
# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

def sigmoid(x):
    sigm = 1 / (1 + np.exp(-x))
    return sigm

