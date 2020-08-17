from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_tests = len(X)
    num_classes = W.shape[1]

    for i in range(num_tests):
      scores = X[i].dot(W)
      C = -np.max(scores)

      exp_scores = np.exp(scores+C)
      exp_scores_sum = np.sum(exp_scores)

      loss += -scores[y[i]]-C+np.log(exp_scores_sum)

      for j in range(num_classes):
        dLdf = exp_scores[j]/exp_scores_sum
        if j == y[i]:
          dLdf -= 1
        
        dW[:, j] += dLdf*X[i]

    loss /= num_tests
    dW /= num_tests
    loss += reg*np.sum(W**2)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_tests = len(y)

    scores = X.dot(W)
    Cs = -np.max(scores, axis=1)
    exp_scores = np.exp(scores+Cs[:,np.newaxis])
    exp_score_sums = np.sum(exp_scores, axis=1)

    losses = -scores[range(num_tests), y]-Cs+np.log(exp_score_sums)
    loss = np.sum(losses)

    dLdf = exp_scores/exp_score_sums[:,np.newaxis]
    dLdf[range(num_tests),y] -= 1
    dW = dLdf.T.dot(X).T

    loss /= num_tests
    loss += reg*np.sum(W**2)
    dW /= num_tests
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW












