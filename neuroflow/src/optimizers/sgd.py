class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]