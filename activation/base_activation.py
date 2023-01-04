from abc import ABC,abstractmethod
import numpy as np

class Activation:
    # methods required in activation
    def __init__(self,activation,d_activation):
        self._input = None
        self._output = None
        self.activation = activation
        self.d_activation = d_activation
        
    def __call__(self,input_tensor:np.ndarray)->np.ndarray:
        ...
  
    def backward_pass(self,output_gradient:np.ndarray,learning_rate:float)\
        ->np.ndarray:
        ...