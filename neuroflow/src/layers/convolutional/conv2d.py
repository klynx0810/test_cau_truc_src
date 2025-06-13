import numpy as np
from neuroflow.src.layers.base import Layer
from typing import Dict
from ....registry import get_activation

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding=0, input_shape=None, activation=None, name=None):
        """
        filters: số lượng filter (số output channels)
        kernel_size: kích thước của mỗi filter (int hoặc tuple)
        stride: bước trượt (default = 1)
        padding: padding (số pixel đệm vào mỗi phía, default = 0)
        input_shape: tuple (h, w, c) nếu biết trước
        activation: tên hàm kích hoạt (vd: 'relu')
        """
        super().__init__(name=name)
        self.filters = filters
        # self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.stride = stride
        self.padding = padding 
        self.input_shape = input_shape  
        self.last_input: np.ndarray = None
        self.params: Dict[str, np.ndarray] = {}
        self.activation: Layer = get_activation(activation) if activation else None

    def build(self, input_shape):
        """
        input_shape: (batch_size, height, width, channels)
        """
        self.batch_size, in_h, in_w, in_c = input_shape
        kh, kw = self.kernel_size
        # khởi tạo W: (filters, kh, kw, in_c), mỗi filter dùng cho mọi kênh
        self.params["W"] = np.random.randn(self.filters, kh, kw, in_c) * 0.01
        # self.params["W"] = np.ones((self.filters, kh, kw, in_c)) * 2
        self.params["b"] = np.zeros((self.filters,))
        self.built = True

    def conv2d_single_channel(self, A, W, bias, stride, pad):
        """
        Hàm hỗ trợ: Tính tích chập cho 1 ảnh 2D và 1 filter 2D
        """
        n_H_old, n_W_old = A.shape
        f, _ = W.shape
        A_pad = np.pad(A, pad_width=pad, mode='constant', constant_values=0)

        n_H_new = int((n_H_old - f + 2 * pad) / stride) + 1
        n_W_new = int((n_W_old - f + 2 * pad) / stride) + 1

        out = np.zeros((n_H_new, n_W_new))
        for i in range(n_H_new):
            for j in range(n_W_new):
                vert_start = i * stride
                horiz_start = j * stride
                out[i, j] = np.sum(A_pad[vert_start:vert_start+f, horiz_start:horiz_start+f] * W) + bias
        return out

    def forward_basic(self, x: np.ndarray):
        """
        x: input đầu vào, shape (batch_size, height, width, channels)
        Trả về: output shape (batch_size, out_h, out_w, filters)
        """
        if not self.built:
            self.build(x.shape)

        self.last_input = x
        B, H, W, C = x.shape
        F, kh, kw, _ = self.params["W"].shape
        stride = self.stride
        pad = self.padding

        out_h = int((H - kh + 2 * pad) / stride) + 1
        out_w = int((W - kw + 2 * pad) / stride) + 1
        out = np.zeros((B, out_h, out_w, F))

        for b in range(B):
            for f in range(F):
                out_channel = np.zeros((out_h, out_w))
                for c in range(C):
                    A = x[b, :, :, c]
                    Wf = self.params["W"][f, :, :, c]
                    out_channel += self.conv2d_single_channel(A, Wf, 0, stride, pad)
                out_channel += self.params["b"][f]
                out[b, :, :, f] = out_channel

        if self.activation:
            output = self.activation.forward(out)
        else:
            output = out

        return output

    def backward_basic(self, grad_output: np.ndarray):
        """
        grad_output: gradient từ layer sau, shape (batch_size, out_h, out_w, filters)
        Trả về: grad_input truyền ngược lại (same shape as input)
        """
        # raise NotImplementedError("Chưa triển khai Conv2D.backward")
        B, H, W, C = self.last_input.shape
        F, kh, kw, _ = self.params["W"].shape
        _, H_out, W_out, _ = grad_output.shape

        dL_dX = np.zeros_like(self.last_input)
        dL_dW = np.zeros_like(self.params["W"])
        dL_db = np.zeros_like(self.params["b"])

        X_padded = np.pad(self.last_input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')
        dL_dX_padded = np.pad(dL_dX, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        for b in range(B):
            for f in range(F):
                for u in range(H_out):
                    for v in range(W_out):
                        # index để cắt vùng
                        vert_start = u * self.stride
                        horiz_start = v * self.stride
                        
                        vert_end = u * self.stride + kh
                        horiz_end = v * self.stride + kw
                        for c in range(C):
                            X_um_vn = X_padded[b, vert_start: vert_end, horiz_start: horiz_end, c]
                            dL_dW[f, :, :, c] += grad_output[b, u, v, f] * X_um_vn
                            dL_dX_padded[b, vert_start: vert_end, horiz_start: horiz_end, c] += grad_output[b, u, v, f] * self.params["W"][f, :, :, c]
                        
                dL_db[f] += np.sum(grad_output[b, :, :, f])

        if self.padding > 0:
            dL_dX = dL_dX_padded[:, self.padding : -self.padding, self.padding : - self.padding, :]
        else:
            dL_dX = dL_dX_padded

        self.grads["W"] = dL_dW
        self.grads["b"] = dL_db

        return dL_dX

    def im2col(self, x: np.ndarray, kh, kw, stride=1, padding=0):
        B, W, H, C = x.shape
        OH = (H + 2 * padding - kh) // stride + 1
        OW = (W + 2 * padding - kw) // stride + 1

        X_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0,0)), mode="constant")
        cols = []

        for i in range(OH):
            for j in range(OW):
                patch = X_pad[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :]
                # (B, kh*kw*C)
                cols.append(patch.reshape(B, -1))

        # (B, OH*OW, kh*kw*C)
        cols = np.stack(cols, axis=1)
        return cols.reshape(B * OH * OW, kh * kw * C), OH, OW
    
    def im2col_optimized(self, x: np.ndarray, kh, kw, stride=1, padding=0):
        B, H, W, C = x.shape
        OH = (H + 2 * padding - kh) // stride + 1
        OW = (W + 2 * padding - kw) // stride + 1

        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
        
        shape = (B, OH, OW, kh, kw, C)
        strides = (
            x_padded.strides[0],
            stride * x_padded.strides[1],
            stride * x_padded.strides[2],
            x_padded.strides[1],
            x_padded.strides[2],
            x_padded.strides[3]
        )
        
        patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        return patches.reshape(B * OH * OW, -1), OH, OW


    def forward(self, x: np.ndarray):
        """
        x: (B, H, W, C)
        kernel: (F, kh, kw, C)
        return: (B, OH, OW, F)
        """
        if not self.built:
            self.build(x.shape)

        self.last_input = x
        
        F, kh, kw, C = self.params["W"].shape
        cols, OH, OW = self.im2col_optimized(x, kh, kw, self.stride, self.padding)     # (B*OH*OW, kh*kw*C)
        W = self.params["W"].reshape(F, -1)                                  # (F, kh*kw*C)
        
        out: np.ndarray = cols @ W.T                                           # (B*OH*OW, F)
        B = x.shape[0]
        out = out.reshape(B, OH, OW, F)

        if self.activation:
            output = self.activation.forward(out)
        else:
            output = out

        return output

    def col2im(self, dX_col: np.ndarray, input_shape, kh, kw, stride=1, padding=0):
        """
        dX_col: (B * OH * OW, kh * kw * C) - gradient ở dạng patch
        Trả về dx: (B, H, W, C) - gradient theo input
        """
        B, H, W, C = input_shape
        OH = (H + 2 * padding - kh) // stride + 1
        OW = (W + 2 * padding - kw) // stride + 1

        dx_padded = np.zeros((B, H + 2 * padding, W + 2 * padding, C))
        dX_col = dX_col.reshape(B, OH * OW, kh * kw * C)

        idx = 0
        for i in range(OH):
            for j in range(OW):
                patch = dX_col[:, idx, :].reshape(B, kh, kw, C)
                dx_padded[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :] += patch
                idx += 1

        if padding > 0:
            return dx_padded[:, padding:-padding, padding:-padding, :]
        return dx_padded
    
    def col2im_optimized(self, dX_col: np.ndarray, input_shape, kh, kw, stride=1, padding=0):
        """
        dX_col: (B * OH * OW, kh * kw * C)
        input_shape: (B, H, W, C)
        """
        B, H, W, C = input_shape
        OH = (H + 2 * padding - kh) // stride + 1
        OW = (W + 2 * padding - kw) // stride + 1
        H_pad, W_pad = H + 2 * padding, W + 2 * padding

        dX_col = dX_col.reshape(B, OH, OW, kh, kw, C)

        dx_padded = np.zeros((B, H_pad, W_pad, C), dtype=dX_col.dtype)

        shape = (B, OH, OW, kh, kw, C)
        strides = (
            dx_padded.strides[0],
            stride * dx_padded.strides[1],
            stride * dx_padded.strides[2],
            dx_padded.strides[1],
            dx_padded.strides[2],
            dx_padded.strides[3],
        )
        dx_strided = np.lib.stride_tricks.as_strided(dx_padded, shape=shape, strides=strides)

        np.add.at(dx_strided, (), dX_col)

        if padding > 0:
            return dx_padded[:, padding:-padding, padding:-padding, :]
        return dx_padded


    def backward(self, grad_output: np.ndarray):
        """
        grad_output: (B, OH, OW, F)
        Trả về: dx: (B, H, W, C)
        """
        if self.activation:
            grad_output = self.activation.backward(grad_output=grad_output)
            
        B, H, W, C = self.last_input.shape
        F, kh, kw, _ = self.params["W"].shape
        _, H_out, W_out, _ = grad_output.shape

        dL_dX = np.zeros_like(self.last_input)
        dL_dW = np.zeros_like(self.params["W"])
        # dL_db = np.zeros_like(self.params["b"])

        cols, OH, OW = self.im2col_optimized(self.last_input, kh, kw, self.stride, self.padding)  # (B*OH*OW, kh*kw*C)
        grad_output_flat = grad_output.reshape(B * OH * OW, F)  # (B*OH*OW, F)

        dL_dW = grad_output_flat.T @ cols
        dL_dW = dL_dW.reshape(F, kh, kw, C)
        dL_db = np.sum(grad_output, axis=(0, 1, 2)) # (F,)

        W_flat = self.params["W"].reshape(F, -1)  # (F, kh*kw*C)
        dL_dX_col = grad_output_flat @ W_flat   # (B*OH*OW, kh*kw*C)
        dL_dX = self.col2im_optimized(dL_dX_col, self.last_input.shape, kh, kw, self.stride, self.padding)  # (B, H, W, C)

        self.grads["W"] = dL_dW
        self.grads["b"] = dL_db

        return dL_dX
    
    def get_config(self):
        base_config:dict = super().get_config()
        base_config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "input_shape": self.input_shape,
            "activation": self.activation.name if self.activation else None
        })
        return base_config
