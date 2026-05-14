import numpy as np
import scipy as sp
import matplotlib.pyplot as mt

class network:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def zed(self, w, b, x):
        return (w * x) + b

    def relu(self, z):
        return max(0, z)

    def forward(self, x, layer_nums=3):
        a = x

        for i in range(layer_nums):

            W = self.weights[i]
            b = self.bias[i]
            z = self.zed(W, b, a)
            a = self.relu(z)
            print(f"Layer {i} | Z = {z}, Relu = {a}")
        
        return self.zed(self.weights[i], self.bias[i], a)


nn = network([2, 1, 0.5], [0.2, 0.5, 1])
data = [(3, 9), (2, 6), (4, 12), (7, 21)]

for i in range(len(data)):
    print(f"-----------------{data[i][0]}----------------")
    nn.forward(data[i][0])