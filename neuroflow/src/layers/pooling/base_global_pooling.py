import numpy as np
from ..base import Layer

class BaseGlobalPooling(Layer):
    def __init__(self, keepdims=False, name=None):
        """
        Khởi tạo lớp global pooling cơ bản.
        keepdims: nếu True thì giữ nguyên chiều không gian (1,1)
        name: tên layer nếu cần (dùng bởi lớp Layer cha)
        """
        super().__init__(name=name)
        self.keepdims = keepdims

    def pool_function(self, region: np.ndarray):
        """
        Hàm pooling trên toàn bộ vùng không gian của mỗi channel.
        region: mảng 2D (H × W) của một channel
        return: giá trị pooling (vd: max, mean)
        """
        raise NotImplementedError("Bạn cần định nghĩa pool_function trong lớp con")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Thực hiện global pooling trên toàn bộ chiều không gian (H, W).
        x: tensor đầu vào (B, H, W, C)
        return:
            - Nếu keepdims=False → shape (B, C)
            - Nếu keepdims=True  → shape (B, 1, 1, C)
        """
        B, H, W, C = x.shape

        if self.keepdims:
            out = np.zeros((B, 1, 1, C))
            for b in range(B):
                for c in range(C):
                    out[b, 0, 0, c] = self.pool_function(x[b, :, :, c])
            return out
        else:
            return np.array([
                [self.pool_function(x[b, :, :, c]) for c in range(C)]
                for b in range(B)
            ])
