import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

# copied from cnn.py and modify on it
class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # conv layer
    # (C, H, W) -> (num_filters, Hc, Wc)
    C, H, W = input_dim
    W1_dim = (num_filters, C, filter_size, filter_size)
    W1 = weight_scale * np.random.randn(*W1_dim)
    b1 = np.zeros(num_filters)
    
    pad = (filter_size - 1) / 2
    stride = 1
    Hc = (H - filter_size + 2 * pad) / stride + 1
    Wc = (W - filter_size + 2 * pad) / stride + 1

    # max pool layer (2 * 2)
    # (num_filters, Hc, Hc) -> (num_filters, Hp, Hp)
    Hp = Hc / 2

    # FC hidden layer
    # (num_filters, Hp, Hp) -> hidden_dim
    fc_dim = num_filters * Hp * Hp
    W2 = weight_scale * np.random.randn(fc_dim, hidden_dim)
    b2 = np.zeros(hidden_dim)

    # output FC layer
    # hidden_dim -> num_classes
    W3 = weight_scale * np.random.randn(hidden_dim, num_classes)
    b3 = np.zeros(num_classes)

    # update self.params
    self.params.update({"W1": W1, "b1": b1,
                        "W2": W2, "b2": b2,
                        "W3": W3, "b3": b3})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
    relu1, relu1_cache = relu_forward(conv)
    mpool, mp_cache = max_pool_forward_fast(relu1, pool_param)
    fc1, fc1_cache = affine_forward(mpool, W2, b2)
    relu2, relu2_cache = relu_forward(fc1)
    scores, fc2_cache = affine_forward(relu2, W3, b3) 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    grads = {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss = data_loss + reg_loss

    drelu2, dW3, db3 = affine_backward(dscores, fc2_cache)
    dfc1 = relu_backward(drelu2, relu2_cache)
    dmpool, dW2, db2 = affine_backward(dfc1, fc1_cache)
    drelu1 = max_pool_backward_fast(dmpool, mp_cache)
    dconv = relu_backward(drelu1, relu1_cache)
    dx, dW1, db1 = conv_backward_fast(dconv, conv_cache)

    grads.update({"W3": dW3, "b3": db3,
                  "W2": dW2, "b2": db2,
                  "W1": dW1, "b1": db1})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads