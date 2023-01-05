
class Metrics():
    
    def __init__(self,metric):
        self._metric = metric
    
    def __call__(self,y_true,y_pred):
        return self._metric(y_true,y_pred)