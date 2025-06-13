import numpy as np
from ..base import Layer
from ....registry import get_activation

class Dense(Layer):
    def __init__(self, units, input_dim=None, activation=None, name=None):
        """
        units: số lượng node ở layer này (số chiều output)
        input_dim: số chiều của input vector (nếu biết trước)
        activation: tên hàm kích hoạt (vd: 'relu', 'tanh', ...)
        name: tên lớp (tuỳ chọn)
        """
        super().__init__(name=name)
        self.units = units
        self.input_dim = input_dim
        self.activation: Layer = get_activation(activation) if activation else None

    def build(self, input_shape):
        """
        input_shape: tuple, thường là (batch_size, input_dim)
        """
        input_dim = self.input_dim or input_shape[-1]
        self.params["W"] = np.random.randn(input_dim, self.units) * 0.01
        self.params["b"] = np.zeros((self.units,))
        self.built = True

    def forward(self, x: np.ndarray):
        """
        x: đầu vào có shape (batch_size, input_dim)
        Trả về: output shape (batch_size, units)
        """
        if not self.built:
            self.build(x.shape)
        self.last_input = x
        W = self.params["W"]
        b = self.params["b"]
        
        output = x @ W + b
        if self.activation:
            output = self.activation.forward(output)

        return output
    
    def backward(self, grad_output: np.ndarray):
        """
        grad_output: gradient từ layer phía sau (shape: [batch_size, units])
        Trả về: grad_input (shape: [batch_size, input_dim])
        """
        if self.activation:
            grad_output = self.activation.backward(grad_output=grad_output)
            
        W = self.params["W"]
        x = self.last_input

        # Gradient w.r.t weights and bias
        self.grads["W"] = x.T @ grad_output      # shape: (input_dim, units)
        self.grads["b"] = np.sum(grad_output, axis=0)  # shape: (units,)

        # Gradient w.r.t input (truyền cho layer trước đó)
        grad_input = grad_output @ W.T           # shape: (batch_size, input_dim)
        return grad_input

    def get_config(self):
        base_config: dict = super().get_config()
        base_config.update({
            "units": self.units,
            "activation": self.activation.name,
            "input_dim": self.input_dim
        })
        return base_config
