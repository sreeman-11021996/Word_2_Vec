import numpy as np
from layers.base_layer import Layer

class Dense(Layer):
    SGD = "sgd"
    HE_NORM = "he_normal"
    GLORAT_NORM = "glorat_normal"

    def __init__ (self,units:int,weight_kernel="he_normal"):
        """
        attributes:
        units -> number of nodes in the layer
        weight_kernel = "he_normal","glorat_normal"
        """
        # input & output
        self._units = units         # output_dim
        self._input_units = None    # input_dim
        
        # wts and bias
        self._weights:np.ndarray = None
        self._bias:np.ndarray = None
        self._d_w:np.ndarray = None
        self._d_b:np.ndarray = None
        
        # optimizations
        self._weight_kernel:str = weight_kernel
        self._optimizer:str = Dense.SGD

    # d_w and d_b Getter 
    @property
    def grad_weights(self):
        return self._d_w
  
    @property
    def grad_bias(self):
        return self._d_b

    # Wts and Bias Getter
    @property
    def weights(self):
        return self._weights
  
    @property
    def bias(self):
        return self._bias
    
    # Output Getter
    @property
    def output(self):
        return self._output

    # Optimizer getter and setter
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer:str):
        self._optimizer = optimizer

  
    def weight_kernel(self)->float:
        if self._weight_kernel == Dense.HE_NORM:
            std = np.sqrt(2.0/(self._input_units))
        elif self._weight_kernel == Dense.GLORAT_NORM:
            std = np.sqrt(2.0/(self._input_units + self._units))
        return std

    
    def build(self,input_tensor:np.ndarray):
        """
        weights = (input_dim x number_of_nodes)
        bias = (number_of_nodes x 1)
        """
        self._input_units = input_tensor.shape[0]
        self._weights = np.random.randn(self._input_units,self._units)*self.weight_kernel()
        self._bias = np.zeros((self._units,1))

    def __call__ (self,input_tensor:np.ndarray)->np.ndarray:
        # Forward Propogation
        """
        Z = np.dot(W.T,X) + b
        W.T -> (number_of_nodes x input_dim)
        X   -> (input_dim x number_of_samples)
        b   -> (number_of_nodes x 1)
        Therefore - Z   -> (number_of_nodes x number_of_samples)
        """
        if self._weights is None:
            self.build(input_tensor=input_tensor)

        self._input = input_tensor
        self._output = np.dot(np.transpose(self._weights),input_tensor) + \
            self._bias
        return self._output
    
   
    def backward_pass(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Backward Propogation
        """
        => d_w_l = A(l-1).(d_L/d_A(l)).T
        gradient weights of layer l : d_w_l
        output_gradient.T [(d_L/d_A(l)).T]: (number_of_examples x number_of_nodes)
        _input [A(l-1)]: (input_dim x number_of_examples)
            
        => d_L/d_A(l-1) = w_l.(d_L/d_A(l))
        input_gradient    : (input_dim x number_of_examples)
        w_l               : (input_dim x number_of_nodes)
        """
        self.grad_weights = np.dot(self._input,np.transpose(output_gradient))
        input_gradients = np.dot(self._weights,output_gradient)
        self.update(learning_rate=learning_rate)
        return input_gradients

    def update (self, learning_rate:float):
        if self._optimizer == Dense.SGD:
            self._sgd(learning_rate=learning_rate)

    def _sgd(self, learning_rate:float):
        self._weights = self._weights + learning_rate*self._d_w
        self._bias = self._bias + learning_rate*self._d_b