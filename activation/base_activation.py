from abc import ABC,abstractmethod
import numpy as np

class Base_Activation(ABC):
    
    @abstractmethod
    def __call__(self,input_tensor:np.ndarray)->np.ndarray:
        ...
        
    @abstractmethod
    def backward_pass(self,output_gradient:np.ndarray)->np.ndarray:
        ...
        
class Activation(Base_Activation):
    # methods required in activation
    def __init__(self,activation,d_activation):
        self._input = None
        self._output = None
        self.activation = activation
        self.d_activation = d_activation
        
    def __call__(self,input_tensor:np.ndarray)->np.ndarray:
        self._input = input_tensor
        self._output = self.activation(self._input)
        return self._output
  
    def backward_pass(self,output_gradient:np.ndarray)->np.ndarray:
        return np.multiply(output_gradient, self.d_activation(self._input))