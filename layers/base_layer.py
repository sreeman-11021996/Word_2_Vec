from abc import ABC,abstractmethod
import numpy as np

class Layer(ABC):
    
    @abstractmethod
    def __init__(self):
        self._input = None
        self._output = None
    
    @property
    @abstractmethod
    def output(self):
        ...
  
    @abstractmethod
    # Forward Propogation
    def __call__ (self,input_tensor:np.ndarray)->np.ndarray: 
        ...

    @abstractmethod
    def build (self,input_tensor:np.ndarray):
        ...
    
    @abstractmethod
    def backward_pass(self,output_gradient:np.ndarray,\
        learning_rate:float)->np.ndarray:
        ...
  
    @abstractmethod
    def update (self, learning_rate:float):
        ...
        
    
  