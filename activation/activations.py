from activation.base_activation import Activation,Base_Activation
import numpy as np

class Sigmoid(Activation):
    
    def __init__(self):
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def d_sigmoid(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid,d_sigmoid)

class Tanh(Activation):
    
    def __init__(self):
        
        def tanh(x):
            return np.tanh(x)
        
        def d_tanh(x):
            h = tanh(x)
            return 1 - h ** 2
        
        super().__init__(tanh,d_tanh)

class ReLU(Activation):
    
    def __init__(self):
        
        def relu(x):
            return np.maximum(x,0)
                    
        def d_relu(x:np.ndarray)->np.ndarray:
            result = x.copy()
            result[x >= 0] = 1
            result[x < 0] = 0
            return result
            
        super().__init__(relu,d_relu)
        
class Linear(Activation):
    
    def __init__(self):
        
        def linear(x):
            return x
                    
        def d_linear(x:np.ndarray)->np.ndarray:
            return np.ones(x.shape)
            
        super().__init__(linear,d_linear)
        
class Softmax(Base_Activation):
        
    def __call__(self,input_tensor:np.ndarray):
        ...
        
    def backward_pass(self, output_gradient:np.ndarray)->np.ndarray:
        ...
