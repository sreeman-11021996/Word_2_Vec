from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

    @abstractmethod
    def __call__(self,y_pred:np.ndarray,y_true:np.ndarray):
        ...

    @abstractmethod
    def gradient(self,y_pred:np.ndarray,y_true:np.ndarray)-> np.ndarray:
        ...