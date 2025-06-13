import numpy as np
from ..base import Layer

class ReLU(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output * (self.input > 0)
        return grad_input
    
    def get_config(self):
        return super().get_config()