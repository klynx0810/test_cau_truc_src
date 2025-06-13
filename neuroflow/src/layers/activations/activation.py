from ..base import Layer
from ....registry import get_activation

class Activation(Layer):
    def __init__(self, activation, name=None):
        super().__init__(name=name)
        self.activation: Layer = get_activation(activation)

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, grad_output):
        return self.activation.backward(grad_output)
    
    def get_config(self):
        return super().get_config()
