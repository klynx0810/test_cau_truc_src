class Layer:
    def __init__(self, name=None, trainable=True) -> None:
        """
        name: tên lớp (nếu không truyền thì tự động lấy tên class viết thường)
        trainable: lớp có tham số cần học (W, b) hay không
        """
        self.name = name or self.__class__.__name__.lower()
        self.trainable = trainable
        self.built = False
        self.params = {}
        self.grads = {}

    def build(self, input_shape):
        """
        Mặc định không làm gì. Các lớp con override build(input_shape)
        để khởi tạo W, b dựa trên shape của input.
        """
        pass

    def forward(self, x):
        """Tính toán đầu ra từ đầu vào"""
        raise NotImplementedError("Layer phải định nghĩa forward()")

    def backward(self, grad_output):
        """Lan truyền gradient ngược lại"""
        raise NotImplementedError("Layer phải định nghĩa backward()")

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
    
    def get_config(self):
        """Trả về cấu hình layer, override ở lớp con nếu cần"""
        return {
            "name": self.name,
            # "trainable": self.trainable
        }