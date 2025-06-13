import numpy as np
from ..base import Layer

class Softmax(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Hàm softmax ổn định về số học
        x: (batch_size, num_classes)
        """
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))  # trừ max tránh tràn số
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Tính output softmax
        x: (batch_size, num_classes)
        return: (batch_size, num_classes)
        """
        self.output = self._softmax(x)
        return self.output

    def backward_basic(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Tính đạo hàm lan truyền ngược
        grad_output: (batch_size, num_classes)
        return: (batch_size, num_classes)
        """
        dx = np.empty_like(grad_output)
        for i in range(len(grad_output)):
            y = self.output[i].reshape(-1, 1)  # (C, 1)
            jacobian = np.diagflat(y) - y @ y.T  # (C, C)
            dx[i] = jacobian @ grad_output[i]
        return dx
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Tính đạo hàm lan truyền ngược một cách vector hóa
        grad_output: (B, C)
        return: (B, C)
        """
        y = self.output  # (B, C)
        dot = np.sum(grad_output * y, axis=1, keepdims=True)  # (B, 1)
        return y * (grad_output - dot)
    
    def get_config(self):
        return super().get_config()
