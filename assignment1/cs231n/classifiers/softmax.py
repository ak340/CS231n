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
  num_train = X.shape[0]
  num_class = W.shape[1]
  dprobs = np.zeros((num_train,num_class))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    score = X[i].dot(W)
    score -= np.max(score)
    dn = 0.0
    for j in xrange(num_class):
        score_exp = np.exp(score[j])
        dn += score_exp
        if j==y[i]:
            nm = score_exp
    probs = nm/dn
    loss += -np.log(probs)
    dprobs[i,:] = probs
    dprobs[i,y[i]] -= 1
  print (dprobs.shape,dW.shape)
  dW = X.T.dot(dprobs)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5*reg* np.sum(W*W)
  
  # Gradient regularization
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
  probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
  loss = -np.sum(np.log(probs[np.arange(num_train),y]))

  # Divide the loss by the number of trainig examples
  loss /= num_train
  # Add regularization
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dprobs = probs
  dprobs[np.arange(num_train),y] -= 1
  dW = X.T.dot(dprobs)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

