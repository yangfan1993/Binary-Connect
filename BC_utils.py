import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import nonlinearities

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

# The binarization function
# notice here that although the authors use a variable 'H' corresponds 
# to the value of binarization (-H or H), here we always set H to 1.

def binarization(W, binary=True, deterministic=False, stochastic=False, srng=None):
    """
    The binarization function. This function binarize the gradienta according to different scheme.
    
    Parameters
    ----------
    W : theano.tensor type variable
        The variable to be binarized.
    binary : boolean
        To binarize the gradient or not. Default is True.
    deterministic : boolean
        To control the training/validation process. If True, then do nothing with the gradient. 
        During training, deterministic is always set to False. Default is False.
    stochastic : boolean
        To use stochastic binarization or not. Default is False.
    srng : theano.RandomStreams
        The RandomStreams initializer for the stochastic scheme. Default is None.
    
        
    Returns
    -------
    Wb : The binarized W.
    """
    
    # if not using binary here, don't change W
    # according to the paper, for the stochastic form of BinaryConnect,
    # test-time inference uses the real-valued weights.
    # Hence, if deterministic (i.e. test-time) and stochastic,
    # we should not binarize W.
    if not binary or (deterministic and stochastic):
        Wb = W
    
    else:
        # First apply hard sigmoid
        # Notice: cannot use the hard_sigmoid function in Theano!!!!!
        def hard_sigmoid(x):
            return T.clip((x + 1.) / 2, 0., 1.)
        
        Wb = hard_sigmoid(W)
        
        if stochastic:
            # a binomial distribution of probability Wb
            if not srng:
                srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147483647))
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        else:
            # If deterministic, round to the nearest number
            Wb = T.round(Wb)
        
        # clip Wb to -1 or 1
        Wb = T.cast(T.switch(Wb, 1., -1.), theano.config.floatX)
    
    return Wb



def calculate_update(network, cost, learning_rate, method='adam', binary=True):
    """
    Calculate the update of all parameters in a network w.r.t. 
    the cost function.
    Use adam algorithm with default setting.
    
    Parameters
    ----------
    network : a  list of class:`Layer` instances
        The network to be used
    cost : Theano expression
        The cost function of the network
    learning_rate : float
        Learning rate of the network
    binary : boolean
        If the network contains binarization or not. Default is true
    
    Returns
    -------
    gradient : The gradient of the binarized parameters.
    """
    output_layer = network['out']
    if binary:
        params_W = lasagne.layers.get_all_params(output_layer, binary=True)
        layers = lasagne.layers.get_all_layers(output_layer)
        gradient_binary = []
        for layer in layers:
            # for each layer, we need to calculate the gradient of cost w.r.t Wb instead of W,
            # since only Wb involves the calculation of the final cost.
            params_binary = layer.get_params(binary=True)
            if params_binary:
                gradient_binary.append(theano.grad(cost, layer.Wb))

        # normal params
        params_normal = lasagne.layers.get_all_params(output_layer, trainable=True, binary=False)
        
        # calculate update
        if method == 'adam':
            update_binary = lasagne.updates.adam(gradient_binary, params_W, learning_rate=learning_rate)
            # for other params, use adam directly
            update_normal = lasagne.updates.adam(cost, params_normal, learning_rate=learning_rate)
        elif method == 'sgd':
            update_binary = lasagne.updates.sgd(gradient_binary, params_W, learning_rate=learning_rate)
            # for other params, use sgd directly
            update_normal = lasagne.updates.sgd(cost, params_normal, learning_rate=learning_rate)
        
        # according to the paper, need to clip the gradient of the binaried params
        # Notice: the original implementation do the clipping by scanning through
        # all the layers. I tried to do it by extract all params of the entire
        # network together, but it sometimes fails. Interesting.
        for layer in layers:
            params_binary = layer.get_params(binary=True)
            for param in params_binary:
                update_binary[param] = T.clip(update_binary[param], -1., 1.)
        
        # combine them
        update = OrderedDict(update_binary.items() + update_normal.items())
    
    else:
        # use adam to all params
        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        
        if method == 'adam':
            update = lasagne.updates.adam(cost, params, learning_rate=learning_rate)
        elif method == 'sgd':
            update = lasagne.updates.sgd(cost, params, learning_rate=learning_rate)
                
    return update
    
