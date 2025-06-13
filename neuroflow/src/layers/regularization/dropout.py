import numpy as np
from ..base import Layer

class Dropout(Layer):
    """
    Lớp Dropout dùng để tắt ngẫu nhiên một phần các neuron trong quá trình huấn luyện.

    Parameters:
    -----------
    rate : float
        Tỷ lệ phần tử bị dropout (giá trị từ 0.0 đến dưới 1.0).
    seed : int
        Seed cho bộ sinh số ngẫu nhiên để đảm bảo tái tạo được kết quả.
    name : str
        Tên lớp (nếu không truyền sẽ tự sinh từ tên class).
    """
    def __init__(self, rate=None, seed=42, name=None):
        super().__init__(name=name)
        assert 0.0 <= rate < 1.0, "Tỷ lệ phải nằm trong khoảng [0, 1)"
        self.rate = rate
        self.seed = seed
        self.training = True

    def forward(self, x: np.ndarray):
        """
        Lan truyền xuôi: Áp dụng dropout nếu đang ở chế độ huấn luyện.

        Parameters:
        -----------
        x : np.ndarray
            Đầu vào có shape bất kỳ.

        Returns:
        --------
        np.ndarray
            Đầu ra sau khi đã áp dụng dropout (nếu training=True).
        """
        if self.training:
            np.random.seed(self.seed)
            self.mask = np.random.binomial(n=1, p=(1.0 - self.rate), size=x.shape)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output: np.ndarray):
        """
        Lan truyền ngược: Truyền gradient ngược qua các neuron không bị dropout.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient từ lớp phía sau truyền về.

        Returns:
        --------
        np.ndarray
            Gradient đầu ra sau khi đã nhân với mask dropout.
        """
        return grad_output * self.mask
    
    def get_config(self):
        base_config: dict = super().get_config()
        base_config.update({
            "rate": self.rate,
            "seed": self.seed
        })
        return base_config
