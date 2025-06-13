import numpy as np

class CategoricalCrossentropy:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        """
        y_true: (B, C), one-hot
        y_pred: (B, C), output đã qua softmax
        """
        eps = 1e-12
        # Vector hóa: chỉ cần lấy log prob tại class đúng
        # axis=1: lấy chỉ số class thật (np.argmax dùng được vì one-hot)
        log_probs = np.log(np.clip(y_pred, eps, 1.0))  # tránh log(0)
        loss = -np.sum(y_true * log_probs) / y_true.shape[0]
        return loss

    def backward(self, y_true, y_pred):
        """
        Gradient: dL/dy_pred
        """
        eps = 1e-12
        return -y_true / np.clip(y_pred, eps, 1.0) / y_true.shape[0]
