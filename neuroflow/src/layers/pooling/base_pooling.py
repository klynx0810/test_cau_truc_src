import numpy as np
from ..base import Layer

class BasePooling(Layer):
    def __init__(self, pool_size=None, strides=None, padding="valid", name=None):
        """
        Khởi tạo lớp pooling cơ bản.
        pool_size: tuple (kh, kw)
        strides: tuple (sh, sw). Nếu None thì mặc định bằng pool_size
        padding: 'valid' (không thêm) hoặc 'same' (thêm để giữ kích thước)
        """
        super().__init__(name=name)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()
        assert self.padding in ["valid", "same"], "padding phải là 'valid' hoặc 'same'"

    def pool_function(self, region: np.ndarray):
        """
        Hàm áp dụng trên mỗi vùng pooling.
        region: vùng con 2D trên từng channel
        return: giá trị sau khi pooling (vd: max, mean)
        """
        raise NotImplementedError("Bạn cần định nghĩa pool_function trong lớp con")

    def _pad_input(self, x: np.ndarray):
        """
        Thêm padding nếu padding='same'
        x: tensor đầu vào (B, H, W, C)
        return: tensor đã pad
        """
        B, H, W, C = x.shape
        kh, kw = self.pool_size
        sh, sw = self.strides

        if self.padding == "same":
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))

            pad_h = max((out_h - 1) * sh + kh - H, 0)
            pad_w = max((out_w - 1) * sw + kw - W, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = np.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant"
            )
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Thực hiện pooling trên toàn bộ đầu vào.
        x: tensor đầu vào (B, H, W, C)
        return: tensor đầu ra (B, H_out, W_out, C)
        """
        B, H, W, C = x.shape
        kh, kw = self.pool_size
        sh, sw = self.strides

        x = self._pad_input(x)
        H_padded, W_padded = x.shape[1], x.shape[2]

        out_h = (H_padded - kh) // sh + 1
        out_w = (W_padded - kw) // sw + 1
        out = np.zeros((B, out_h, out_w, C))

        for b in range(B):
            for h in range(out_h):
                for w in range(out_w):
                    for c in range(C):
                        region = x[b,
                                   h * sh:h * sh + kh,
                                   w * sw:w * sw + kw,
                                   c]
                        out[b, h, w, c] = self.pool_function(region)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Hàm lan truyền ngược. Phải được định nghĩa ở lớp con.
        grad_output: gradient từ layer sau
        """
        raise NotImplementedError(f"{self.__class__.__name__} phải định nghĩa backward()")
    
    def get_config(self):
        base_config: dict = super().get_config()
        base_config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return base_config