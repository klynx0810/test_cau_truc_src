import numpy as np

class Accuracy:
    def __init__(self):
        self.name = self.__class__.__name__
        
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) hoặc (batch_size, num_classes) — one-hot hoặc label
        y_pred: (batch_size,) hoặc (batch_size, num_classes) — xác suất hoặc nhãn dự đoán
        """
        if y_pred.ndim > 1:
            y_pred_labels = np.argmax(y_pred, axis=-1)
        else:
            y_pred_labels = (y_pred >= 0.5).astype(int)

        if y_true.ndim > 1:
            y_true_labels = np.argmax(y_true, axis=-1)
        else:
            y_true_labels = y_true

        return np.mean(y_true_labels == y_pred_labels)

    def backward(self, y_true, y_pred):
        """
        Accuracy không có gradient nên trả về None.
        """
        return None
