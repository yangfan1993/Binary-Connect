import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import nonlinearities

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from BC_utils import binarization

# According to the original implementation, we also extend two
# Lasagne layer classes to support BinaryConnect.
# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    """
    Extension to the fully connected layer in Lasagne.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    binary : boolean
        To binarize the weight matrix or not. Default is True
    stochastic : boolean
        To use stochastic binarization or not. Default is True
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. Default is ReLU
    """
    def __init__(self, incoming, num_units, 
                 binary=True, stochastic=True, 
                 nonlinearity=nonlinearities.rectify, 
                 **kwargs):
        
        # An interesting issue here:
        # If I directly call super(), when I define the function for calculating
        # the error when deterministic=True, an TypeError occurs saying that 
        # super(type, obj): obj must be an instance or subtype of type.
        # So here I need to call the super() before init, to make sure
        # the loading of the classes are in the correct order.
        self.as_super = super(DenseLayer, self)
        
        if binary:
            self.as_super.__init__(incoming, num_units, 
                                             W=lasagne.init.Uniform((-1, 1)), 
                                             nonlinearity=nonlinearity, 
                                             **kwargs)
        else:
            self.as_super.__init__(incoming, num_units, 
                                             nonlinearity=nonlinearity, 
                                             **kwargs)
            
        self.binary = binary
        self.stochastic = stochastic
        
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))
        
        if self.binary:
            # This matrix does not involve the calculation of gradient,
            # So here add a tag 'binary' and remove the tag 'trainable'.
            self.params[self.W] = {'binary'}
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        # binarize the weight matrix
        self.Wb = binarization(self.W, self.binary, deterministic, 
                               self.stochastic)
        
        W_real = self.W
        
        # update self.W to the binarized version
        self.W = self.Wb
        
        # use the binarized weight matrix, get the output of this layer
        output = self.as_super.get_output_for(input, **kwargs)
        
        # use the original float weight matrix
        self.W = W_real
        
        return output
    
    
    
# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    """
    Extension to the 2D Convolution layer in Lasagne.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_filters : int
        The number of learnable convolutional filters this layer has
    filter_size : int
        An integer or a 2-element tuple specifying the size of the filters
    binary : boolean
        To binarize the weight matrix or not. Default is True
    stochastic : boolean
        To use stochastic binarization or not. Default is True
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. Default is ReLU
    """
    def __init__(self, incoming, num_filters, filter_size, 
                 binary=True, stochastic=True, 
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        self.as_super = super(Conv2DLayer, self)
        
        if binary:
            self.as_super.__init__(incoming, num_filters, filter_size, 
                                          W=lasagne.init.Uniform((-1., 1.)), 
                                          nonlinearity=nonlinearity, 
                                          **kwargs)
        else:
            self.as_super.__init__(incoming, num_filters, filter_size, 
                                          nonlinearity=nonlinearity, 
                                          **kwargs)
        
        self.binary = binary
        self.stochastic = stochastic
            
        if self.binary:
            # This matrix does not involve the calculation of gradient,
            # So here add a tag 'binary' and remove the tag 'trainable'.           
            self.params[self.W] = {'binary'}
    
    def convolve(self, input, deterministic=False, **kwargs):
        # similar to the get_output function in DenseLayer
        # binarize the weight matrix
        self.Wb = binarization(self.W, self.binary, deterministic, 
                               self.stochastic)
        W_real = self.W
        
        # update self.W to the binarized version
        self.W = self.Wb

        # use the binarized weight matrix, get the output of this layer
        conved = self.as_super.convolve(input, **kwargs)
        
        # use the original float weight matrix
        self.W = W_real
        
        return conved
