"""
Disclaimer: this code is provided as is,

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation;version 2

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

# This is a python version of
# CS294A/CS294W Programming Assignment Starter Code
from sampleIMAGES import sampleIMAGES
import pylab as plt

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  programming assignment. You will need to complete the code in sampleIMAGES.m,
#  sparseAutoencoderCost.m and computeNumericalGradient.m.
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file.
#
##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters you do not need to
#  change the parameters below.

import numpy as np
randi = lambda b,c,d: np.random.randint(0, b, (c, d)) # create a python substitute to the randi function
size = lambda a,b: np.shape(a)[b-1] # create a python substitute to the size function in matlab

visibleSize = 8*8   # number of input units
hiddenSize = 25     # number of hidden units
sparsityParam = 0.01   # desired average activation of the hidden units.
                # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                #  in the lecture notes).
lambdaval = 0.0001     # weight decay parameter
beta = 3            # weight of sparsity penalty term

##======================================================================
## STEP 1: Implement sampleIMAGES
#
#  After implementing sampleIMAGES, the display_network command should
#  display a random sample of 200 patches from the dataset
from display_network import display_network
patches = sampleIMAGES()
display_network(patches[:, randi(size(patches, 2), 200, 1).squeeze() ])
plt.show()

#  Obtain random parameters theta

from initializeParameters import initializeParameters
theta = initializeParameters(hiddenSize, visibleSize)

##======================================================================
## STEP 2: Implement sparseAutoencoderCost
#
#  You can implement all of the components (squared error cost, weight decay term,
#  sparsity penalty) in the cost function at once, but it may be easier to do
#  it step-by-step and run gradient checking (see STEP 3) after each step.  We
#  suggest implementing the sparseAutoencoderCost function using the following steps:
#
#  (a) Implement forward propagation in your neural network, and implement the
#      squared error term of the cost function.  Implement backpropagation to
#      compute the derivatives.   Then (using lambdaval=beta=0), run Gradient Checking
#      to verify that the calculations corresponding to the squared error cost
#      term are correct.
#
#  (b) Add in the weight decay term (in both the cost function and the derivative
#      calculations), then re-run Gradient Checking to verify correctness.
#
#  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
#      verify correctness.
#
#  Feel free to change the training settings when debugging your
#  code.  (For example, reducing the training set size or
#  number of hidden units may make your code run faster and setting beta
#  and/or lambdaval to zero may be helpful for debugging.)  However, in your
#  final submission of the visualized weights, please use parameters we
#  gave in Step 0 above.

from sparseAutoencoderCost import sparseAutoencoderCost
[cost, grad] = sparseAutoencoderCost(theta=theta, visibleSize=visibleSize,
                                     hiddenSize=hiddenSize, lam=lambdaval,
                                     sparsityParam=sparsityParam, beta=beta,
                                     data=patches)
# ======================================================================
#  STEP 3: Gradient Checking
#
# Hint: If you are debugging your code, performing gradient checking on smaller
# models and smaller training sets (e.g., using only 10 training examples and
# 1-2  hidden units) may speed things up.

# First, lets make sure your numerical gradient computation is correct for a
# simple function.  After you have implemented computeNumericalGradient.m,
# run the following:

from checkNumericalGradient import checkNumericalGradient
checkNumericalGradient()

# Now we can use it to check your cost function and derivative calculations
# for the sparse autoencoder.
from computeNumericalGradient import computeNumericalGradient
numgrad = computeNumericalGradient(lambda x:sparseAutoencoderCost(x, visibleSize=visibleSize,
                                                  hiddenSize=hiddenSize, lam=lambdaval,
                                                  sparsityParam=sparsityParam, beta=beta,
                                                  data=patches)[0], theta)

# Use this to visually compare the gradients side by side
print [numgrad, grad]

# Compare numerically computed gradients with the ones obtained from backpropagation
diff = np.linalg.norm(numgrad-grad)*1.0/np.linalg.norm(numgrad+grad)
print 'the difference in the gradients is: ', diff
# print diff # Should be small. In our implementation, these values are
            # usually less than 1e-9.


            # When you got this working, Congratulations!!!

##======================================================================
## STEP 4: After verifying that your implementation of
#  sparseAutoencoderCost is correct, You can start training your sparse
#  autoencoder with minFunc (L-BFGS).

#  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize)


#  Now having verified the correctness of the gradients
# minimize the function using lbfgs
from scipy.optimize import fmin_l_bfgs_b
maxIter = 400	  # Maximum number of iterations of L-BFGS to run
display = True


[opttheta, cost, dictreturned] = fmin_l_bfgs_b(lambda p: sparseAutoencoderCost(p,
                                                      visibleSize=visibleSize,
                                                      hiddenSize=hiddenSize,
                                                      lam=lambdaval,
                                                      sparsityParam=sparsityParam,
                                                      beta=beta,
                                                      data=patches),
                                 x0=theta, disp= display, maxiter=maxIter)


##======================================================================
## STEP 5: Visualization
W1 = opttheta[0:hiddenSize*visibleSize].reshape([hiddenSize, visibleSize])
plt.figure()
display_network(W1.T)
plt.title('visualization of feature vectors')
plt.show()

