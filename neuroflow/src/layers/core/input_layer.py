import numpy as np
from ..base import Layer

class Input(Layer):
    def __init__(self, input_shape=None, name=None):
        super().__init__(name=name)
        self.input_shape = input_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Hàm lan truyền tiến: lớp Input không thay đổi gì dữ liệu,
        chỉ đơn giản truyền qua input ban đầu.
        
        Parameters:
        - x: mảng đầu vào (ví dụ batch_size x 784)
        
        Returns:
        - chính là x, giữ nguyên không thay đổi
        """
        if self.input_shape is not None:
            expected_shape = (x.shape[0],) + self.input_shape
            if x.shape != expected_shape:
                raise ValueError(
                    f"Kích thước không khớp, mong đợi là {expected_shape}, nhưng truyền vào là {x.shape}"
                    f"[Input] Lỗi kích thước! Dữ liệu đầu vào không đúng.\n"
                    f"  - Đã khai báo input_shape = {self.input_shape}\n"
                    f"  - Kỳ vọng đầu vào có shape = {expected_shape}\n"
                    f"  - Nhưng nhận được = {x.shape}"
                    ) 
        self.output = x
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Hàm lan truyền ngược: không thay đổi gì gradient,
        chỉ truyền gradient từ lớp sau về lớp trước như nguyên trạng.
        
        Parameters:
        - grad_output: gradient từ lớp phía sau
        
        Returns:
        - chính là grad_output, giữ nguyên không thay đổi
        """
        return grad_output
    
    def get_config(self):
        base_config: dict = super().get_config()
        base_config.update({
            "input_shape": self.input_shape
        })
        return base_config