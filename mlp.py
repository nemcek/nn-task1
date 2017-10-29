import numpy as np

from util import *


class MLP():

    def __init__(self, dim_in, dim_hid, dim_out):
        self.dim_in     = dim_in
        self.dim_hid    = dim_hid
        self.dim_out    = dim_out

        self.W_hid = np.random.rand(dim_hid, dim_in + 1)
        self.W_out = np.random.rand(dim_out, dim_hid + 1)


    def forward(self, x):
        a = self.W_hid @ augment(x)
        h = augment(self.f_hid(a))
        b = self.W_out @ h
        y = self.f_out(b)

        return y, b, h, a


    def backward(self, x, d):
        y, b, h, a = self.forward(x)

        g_out = (d - y) * self.df_out(b)
        g_hid = (g_out.T @ self.W_out[:, 0:-1]) * self.df_hid(a)

        dW_out = h.reshape((1, -1)).T @ g_out.reshape((1, -1))
        dW_hid = augment(x).reshape((1, -1)).T @ g_hid.reshape((1, -1))

        return y, dW_hid, dW_out
