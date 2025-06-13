from .model import Model
from ..layers.base import Layer
from typing import List, Optional

class Sequential(Model):
    def __init__(self, layers: Optional[List[Layer]] = None, name=None):
        """
        Sequential: mô hình tuyến tính, các layer xếp nối tiếp nhau.
        
        layers: danh sách các layer sẽ được add() lần lượt.
        name: tên mô hình (tuỳ chọn).
        """
        super().__init__(name=name)
        if layers:
            for layer in layers:
                self.add(layer)
