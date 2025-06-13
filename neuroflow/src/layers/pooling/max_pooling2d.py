import numpy as np
from ..base import Layer
from .base_pooling import BasePooling

class MaxPooling2D(BasePooling, Layer):
    def __init__(self, pool_size=None, strides=None, padding="valid", name=None):
        """
        MaxPooling2D kế thừa BasePooling
        pool_size: tuple (kh, kw)
        strides: tuple (sh, sw), nếu None thì = pool_size
        padding: 'valid' hoặc 'same'
        name: tên lớp nếu cần
        """
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, name=name)

    def pool_function(self, region: np.ndarray):
        """Lấy giá trị lớn nhất trong vùng"""
        return np.max(region)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: tensor đầu vào (B, H, W, C)
        return: output sau pooling (B, H_out, W_out, C)
        """
        self.input = x  # lưu lại input để backward
        return super().forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Lan truyền ngược qua MaxPooling2D
        grad_output: gradient từ lớp sau (B, H_out, W_out, C)
        return: gradient theo input gốc (B, H, W, C)
        """
        B, H, W, C = self.input.shape
        kh, kw = self.pool_size
        sh, sw = self.strides

        x_padded = self._pad_input(self.input)
        dX = np.zeros_like(x_padded)
        H_out, W_out = grad_output.shape[1:3]

        for b in range(B):
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C):
                        h_start = h * sh
                        w_start = w * sw
                        region = x_padded[b, h_start:h_start+kh, w_start:w_start+kw, c]
                        max_val = np.max(region)
                        mask = (region == max_val)
                        dX[b, h_start:h_start+kh, w_start:w_start+kw, c] += grad_output[b, h, w, c] * mask

        # Cắt lại nếu có padding
        if self.padding == "same":
            pad_h = x_padded.shape[1] - H
            pad_w = x_padded.shape[2] - W
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            return dX[:, pad_top:pad_top+H, pad_left:pad_left+W, :]
        return dX
