from activation.base_activation import Activation
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

class Relu(Activation):
    
    def __init__(self):
        
        def relu(x):
            ...
                    
        def d_relu(x):
            ...
            
        super().__init__(relu,d_relu)
