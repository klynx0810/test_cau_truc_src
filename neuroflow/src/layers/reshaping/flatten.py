import numpy as np
from ..base import Layer

class Flatten(Layer):
    def __init__(self, input_shape=None, name=None):
        """
        Lớp Flatten biến đổi đầu vào từ (B, d1, d2, ..., dn) → (B, D)
        name: tên lớp (tuỳ chọn)
        """
        super().__init__(name=name)
        self.input_shape = input_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: input (batch_size, d1, d2, ..., dn)
        return: (batch_size, d1 * d2 * ... * dn)
        """
        self.input_shape = x.shape  # lưu lại để dùng khi backward
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: gradient từ layer sau, shape (batch_size, D)
        return: reshape lại về input shape ban đầu
        """
        return grad_output.reshape(self.input_shape)

    def get_config(self):
        base_config: dict = super().get_config()
        base_config.update({
            "input_shape": self.input_shape
        })
        return base_config
