import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}     # moment bậc 1
        self.v = {}     # moment bậc 2
        self.t = 0      # timestep

    def step(self, params, grads):
        self.t += 1

        for key in params:
            # Khởi tạo m và v nếu chưa có
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])

            # Cập nhật m và v
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Cập nhật tham số
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
