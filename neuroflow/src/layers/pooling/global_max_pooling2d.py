import numpy as np
from ..base import Layer
from .base_global_pooling import BaseGlobalPooling

class GlobalMaxPooling2D(BaseGlobalPooling, Layer):
    def __init__(self, keepdims=False, name=None):
        """
        Lớp GlobalMaxPooling2D kế thừa BaseGlobalPooling
        keepdims: nếu True → giữ shape (B, 1, 1, C); nếu False → (B, C)
        name: tên lớp nếu cần
        """
        super().__init__(keepdims=keepdims, name=name)

    def pool_function(self, region: np.ndarray):
        """Lấy giá trị lớn nhất toàn bộ vùng 2D"""
        return np.max(region)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: tensor đầu vào (B, H, W, C)
        return: đầu ra sau global pooling
        """
        self.input = x  # lưu lại để dùng cho backward
        return super().forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua GlobalMaxPooling2D
        grad_output:
            - nếu keepdims=False → shape (B, C)
            - nếu keepdims=True  → shape (B, 1, 1, C)
        return: gradient theo input ban đầu (B, H, W, C)
        """
        B, H, W, C = self.input.shape
        dX = np.zeros_like(self.input)

        for b in range(B):
            for c in range(C):
                region = self.input[b, :, :, c]
                mask = (region == np.max(region))
                g = grad_output[b, c] if not self.keepdims else grad_output[b, 0, 0, c]
                dX[b, :, :, c] = mask * g
        return dX
