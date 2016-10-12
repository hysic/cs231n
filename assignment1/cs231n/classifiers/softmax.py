import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # calculating the loss
  for i in xrange(num_train):
    scores = X[i].dot(W)  # scores for the ith data, 1 * n_classes
    scores -= np.max(scores)  # maintain numeric stability
    sum_ = np.sum(np.exp(scores))
    softmax = np.exp(scores[y[i]]) / sum_
    loss -= np.log(softmax)
    # calculate dW
    for j in xrange(num_classes):
      p_j = np.exp(scores[j]) / sum_
      dW[:, j] += (p_j - (y[i] == j)) * X[i, :]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)  # add reg term

  dW /= num_train
  dW += reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  # loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  # calculate the loss
  scores = X.dot(W)  # n_samples * n_classes
  scores -= np.max(scores, axis=1).reshape(-1, 1)
  scores_exp = np.exp(scores)
  scores_exp_sum = np.sum(scores_exp, axis=1)
  scores_y = np.choose(y, scores_exp.T)
  loss = -np.log((scores_y/scores_exp_sum)).sum()
  # calculate the gradient
  # Thanks to <https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py>
  p = scores_exp / scores_exp_sum.reshape(-1, 1)
  ind = np.zeros_like(p)
  ind[range(num_train), y] = 1
  dW = np.dot(X.T, (p-ind))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

