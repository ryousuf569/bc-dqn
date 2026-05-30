import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
xs = np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7], dtype=float)
ys = xs ** 2        # square first, on the original values
xs = xs / 7.0       # then normalise
ys = ys / 49.0      # then normalise
xs = xs.reshape(-1, 1)
ys = ys.reshape(-1, 1)

W1 = np.random.randn(8, 1) * np.sqrt(2 / 1)
b1 = np.zeros((1, 8))

W2 = np.random.randn(8, 8) * np.sqrt(2 / 8)
b2 = np.zeros((1, 8))

W3 = np.random.randn(1, 8) * np.sqrt(2 / 8)
b3 = np.zeros((1, 1))

def forward(x, weights):
    
    W1, b1, W2, b2, W3, b3 = weights

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

def backprop(dL_da3, zs, as_, weights):

    W1, b1, W2, b2, W3, b3 = weights
    x,  a1, a2, a3 = as_
    z1, z2, z3 = zs

    # forward had two steps a layer, and we follow the same pattern for backprop
    # we derived these derivatives on paper, let's hardcode them here for now

    dL_dz3 = dL_da3
    dL_dW3 = dL_dz3.T @ a2
    dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

    dL_da2 = dL_dz3 @ W3
    dL_dz2 = np.multiply(dL_da2, np.where(z2 > 0, 1.0, 0.0))
    dL_dW2 = dL_dz2.T @ a1
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

    dL_da1 = dL_dz2 @ W2
    dL_dz1 = np.multiply(dL_da1, np.where(z1 > 0, 1.0, 0.0))
    dL_dW1 = dL_dz1.T @ x
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    return [dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3]

def grad_descent(gradients, weights, lr=0.005):
    W1, b1, W2, b2, W3, b3 = weights
    dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3 = gradients

    W1 = W1 - lr*dL_dW1
    W2 = W2 - lr*dL_dW2
    W3 = W3 - lr*dL_dW3

    b1 = b1 - lr*dL_db1
    b2 = b2 - lr*dL_db2
    b3 = b3 - lr*dL_db3

    return [W1, b1, W2, b2, W3, b3]


def loss(y_hat, ys, n):

    loss = np.sum((y_hat - ys) ** 2) / (2 * n)
    dL_da3 = (y_hat - ys) / n

    return loss, dL_da3

def finite_diff_check(xs, ys, weights, eps=1e-6):
    # run backprop once to get analytic gradients
    y_hat, zs, as_ = forward(xs, weights)
    l, dL_da3 = loss(y_hat, ys, n)
    grads = backprop(dL_da3, zs, as_, weights)
    
    W1, b1, W2, b2, W3, b3 = weights
    analytic_grads = [grads[0], grads[2], grads[4]]  # dW1, dW2, dW3
    weight_mats   = [W1, W2, W3]
    names         = ["W1", "W2", "W3"]
    
    all_passed = True


    # ONLY RUN WHEN NEEDED, triple nested loop will kill if it ran every training loop
    for W, dW, name in zip(weight_mats, analytic_grads, names):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                
                # nudge up
                original = W[i, j]
                W[i, j] = original + eps
                y_hat_up, _, _ = forward(xs, weights)
                loss_up, _ = loss(y_hat_up, ys, n)

                # nudge down
                W[i, j] = original - eps
                y_hat_down, _, _ = forward(xs, weights)
                loss_down, _ = loss(y_hat_down, ys, n)

                # restore
                W[i, j] = original

                # compare
                numeric  = (loss_up - loss_down) / (2 * eps)
                analytic = dW[i, j]
                rel_err  = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-8)

                if rel_err > 1e-4:
                    print(f"FAIL {name}[{i},{j}] | numeric={numeric:.6f} analytic={analytic:.6f} rel_err={rel_err:.2e}")
                    all_passed = False


weights = [W1, b1, W2, b2, W3, b3]
n = len(xs)
n_epochs = 100000

for epoch in range(n_epochs):
    # forward
    y_hat, zs, as_ = forward(xs, weights)
    
    # loss
    l, dL_da3 = loss(y_hat, ys, n)
    
    # backward
    grads = backprop(dL_da3, zs, as_, weights)
    
    # run when needed to check gradients
    # finite_diff_check(xs, ys, weights)

    # update
    weights = grad_descent(grads, weights, lr=0.01)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {l:.4f}")
    
# run forward pass on a fine grid of x values
xs_plot = np.linspace(-10, 10, 200).reshape(-1, 1)
ys_true = xs_plot ** 2
ys_pred, _, _ = forward(xs_plot / 10.0, weights)
ys_pred_real = ys_pred * 100.0

plt.figure(figsize=(8, 5))
plt.plot(xs_plot, ys_true, label="y = x²", color="steelblue")
plt.plot(xs_plot, ys_pred_real, label="network prediction", color="tomato", linestyle="--")
plt.scatter(xs * 7.0, ys * 49.0, label="training points", color="steelblue", zorder=5)
plt.legend()
plt.title("yousuf's neural net: predicted vs true")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()