import numpy as np
from ..base import Layer

class Sigmoid(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    
    def get_config(self):
        return super().get_config()