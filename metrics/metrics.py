from metrics.base_metrics import Metrics
import numpy as np

class Accuracy(Metrics):
    
    def __init__(self):
        
        def accuracy(y_true, y_pred):
            return 1.0 - np.mean(np.abs((y_pred - y_true)))

        super().__init__(accuracy)