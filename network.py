import numpy as np

np.random.seed(42)
xs = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype=float)
xs.reshape(-1, 1) # -1 means "figure it out numpy"
ys = xs ** 2
ys.reshape(-1, 1)

W1 = np.random.randn(8, 1) * np.sqrt(2 / 1)
b1 = np.zeros((8, 1))

W2 = np.random.randn(8, 8) * np.sqrt(2 / 8)
b2 = np.zeros((8, 1))

W3 = np.random.randn(1, 8) * np.sqrt(2 / 8)
b3 = np.zeros((1, 1))

def forward(x):
    
    zs = []     # store z at every layer
    as_ = []    # store a at every layer (including input)
    
    a = x       # input is the first activation
    as_.append(a)
    
    z1 = a @ W1.T + b1
    a1 = np.maximum(0, z1)
    zs.append(z1)
    as_.append(a1)

    z2 = a1 @ W2.T + b2
    a2 = np.maximum(0, z2)
    zs.append(z2)
    as_.append(a2)

    z3 = a2 @ W3.T + b3
    a3 = z3
    zs.append(z3)
    as_.append(a3)
    
    return as_[-1], zs, as_

def loss(y_hat, ys, n):

    loss = np.sum((y_hat - ys) ** 2) / (2 * n)
    dL_da3 = (y_hat - ys) / n

    return loss, dL_da3

y_hat, zs, as_ = forward(xs)
loss(y_hat, ys, 11)
