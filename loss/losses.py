from loss.base_loss import Loss
import numpy as np

# Losses Pending:
# 1. Categorical CrossEntropy
# 2. Mean Absolute Error
# 3. Huber Loss
# 4. Hinge Loss

class BinaryCrossEntropy(Loss):
    """
    y_true , y_pred -> (number_of_classes/output_nodes,number_of_samples)
    Calculate the Binary Cross Entropy Loss 
    i.e. Loss = -1/m * np.sum[(y*log(y_hat) + (1-y)*log(1-y_hat))]
    i.e. gradient = d_Loss/d_y_hat
    """
    def __call__(self,y_pred:np.ndarray,y_true:np.ndarray):
        loss = -1 * (np.mean(y_true*np.log(y_pred) + (1-y_true)*(np.log(1-y_pred))))
        return loss

    def gradient(self,y_pred:np.ndarray,y_true:np.ndarray)-> np.ndarray:
        grad_loss = -1 * ((y_true/y_pred) - ((1-y_true)/(1-y_pred)))
        return grad_loss
    
    
class Mean_Squared_Error(Loss):
    """
    y_true , y_pred -> (number_of_classes/output_nodes,number_of_samples)
    Calculate the Binary Cross Entropy Loss 
    i.e. Loss = -1/m * np.sum[(y - y_hat)^2]
    i.e. gradient = d_Loss/d_y_hat
    """
    def __call__(self,y_pred:np.ndarray,y_true:np.ndarray):
        loss = np.mean(np.square(y_true - y_pred))
        return loss

    def gradient(self,y_pred:np.ndarray,y_true:np.ndarray)-> np.ndarray:
        grad_loss = -2 * (y_true - y_pred)
        return grad_loss
    
