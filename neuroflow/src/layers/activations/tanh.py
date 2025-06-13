import numpy as np
from ..base import Layer

class Tanh(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input
    
    def get_config(self):
        return super().get_config()