import numpy as np

def grad_outer_loss(alpha, lmbd, h, X_n, y_n):
    dim = X_n.shape[1]
    n = X_n.shape[0]

    # transform input to have the outer problem as least squares
    C_lambda_n_inv = np.linalg.pinv((X_n.T @ X_n) / n + lmbd * np.eye(dim, dim))
    z_n = X_n.T @ y_n / n
    X_n_hat = (lmbd / np.sqrt(n)) * (X_n @ C_lambda_n_inv)
    y_n_hat = (1 / np.sqrt(n)) * (y_n - X_n @ C_lambda_n_inv @ z_n)

    return (X_n_hat.T @ (X_n_hat @ h - y_n_hat))/n + alpha * h