import numpy as np

class BinaryCrossentropy:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) hoặc (batch_size, 1) — nhãn 0 hoặc 1
        y_pred: (batch_size,) hoặc (batch_size, 1) — output sigmoid
        """
        eps = 1e-12
        loss = - (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return np.mean(loss)

    def backward(self, y_true, y_pred):
        """
        Gradient của loss theo y_pred
        """
        eps = 1e-12
        grad = - (y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps)
        return grad / y_true.shape[0]
