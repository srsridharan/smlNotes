import numpy as np
import ipdb as pdb
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
    b1 = np.reshape(b1, (-1, 1))
    b2 = np.reshape(b2, (-1, 1))

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
    y = data  # output ref
    a1 = data  # input layer
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = sigmoid(z3)

    rho2hat = np.mean(a2, axis=1)
    cost = np.mean(0.5*(np.linalg.norm(a3-data, axis=0)**2)) + 0.5*lam*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2) + beta * np.sum(sparsityParam* np.log(1.0*sparsityParam/rho2hat) + (1-sparsityParam)*np.log((1.0-sparsityParam)/(1-rho2hat)) )
    delta3 = -(y-a3) * (a3 * (1 - a3))
    delta2 = ( (W2.T.dot(delta3))+ beta*(-(sparsityParam*1.0/rho2hat) + (1-sparsityParam)/(1-rho2hat)).reshape((-1,1))) * (a2*(1-a2))
    # delta1= (W1.T*delta2)*(a1*(1-a1))
    W2grad = (1.0/data.shape[1])*delta3.dot(a2.T) + lam*W2
    b2grad = np.mean(delta3, axis=1)
    W1grad = (1.0/data.shape[1])*delta2.dot(a1.T) + lam *W1
    b1grad = np.mean(delta2, axis=1)











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

