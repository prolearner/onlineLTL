import numpy as np


def grad_outer_loss(alpha, lmbd, h, X_n, y_n, X_val, y_val):
    dim = X_n.shape[1]
    n = X_n.shape[0]
    n_val = X_val.shape[0]

    # transform input to have the outer problem as least squares
    C_lambda_n_inv = np.linalg.pinv((X_n.T @ X_n) + n*lmbd*np.eye(dim, dim))
    X_val_hat = n*lmbd*(X_val @ C_lambda_n_inv)
    y_val_hat = y_val - (X_val_hat @ X_val.T @ y_n)/(n*lmbd)

    return X_val_hat.T @ (X_val_hat @ h - y_val_hat)/n_val + alpha * h


def least_mean_squares(gamma, alpha, lmbd, X, y, X_val, y_val,  data_valid, eval_online=True):
    dim = X[0].shape[1]
    n_tasks = len(X)  # T
    hs = np.zeros((n_tasks+1, dim))

    n_tasks_val = len(data_valid['X_train'])

    mean_square_errors_val = np.zeros((n_tasks+1, n_tasks_val))
    mean_square_errors_val[0] = LTL_evaluation(lmbd=lmbd, h=hs.sum(axis=0), X=data_valid['X_train'],
                                               y=data_valid['Y_train'],
                                               X_test=data_valid['X_test'], y_test=data_valid['Y_test'])

    print(str(0) + '-' + 'mse-eval : ', np.mean(mean_square_errors_val[0]), np.std(mean_square_errors_val[0]))

    for t in range(n_tasks):
        hs[t+1] = hs[t] - gamma * grad_outer_loss(alpha=alpha, lmbd=lmbd, h=hs[t], X_n=X[t], y_n=y[t], X_val=X_val[t],
                                                  y_val=y_val[t])

        if eval_online:
            if data_valid is None:
                data_valid = {'X_train': X, 'Y_train': y, 'X_test': X, 'Y_test': y}

            mean_square_errors_val[t+1] = LTL_evaluation(lmbd=lmbd, h=hs.sum(axis=0)/(t+2),  X=data_valid['X_train'],
                                                         y=data_valid['Y_train'], X_test=data_valid['X_test'],
                                                         y_test=data_valid['Y_test'])

            print(str(t) + '-' + 'mse-eval  : ', np.mean(mean_square_errors_val[t+1]), np.std(mean_square_errors_val[t+1]))

    return hs, mean_square_errors_val


def least_mean_squares_train_only(gamma, alpha, lmbd, X, y, data_valid, eval_online=True):
    return least_mean_squares(gamma, alpha, lmbd, X, y, X, y, data_valid, eval_online)


def tikhonov_mean_solver(lmbd, h, X_n, y_n, verbose=0):
    dim = X_n.shape[1]
    n = X_n.shape[0]

    C_lambda_n_inv = np.linalg.pinv((X_n.T @ X_n) + n*lmbd*np.eye(dim, dim))
    w_h = C_lambda_n_inv @ (X_n.T @ y_n + n*lmbd*h)

    mse = 0.5*np.mean((X_n @ w_h - y_n)**2)
    loss = 0.5*(mse + lmbd*np.sum((w_h - h)**2))

    if verbose > 0:
        print('reg-loss', loss)
        print('mse', mse)

    return w_h


def LTL_evaluation(lmbd, h, X, y, X_test, y_test, verbose=0):
    n_tasks = len(X)  # T

    mean_square_errors = np.zeros(n_tasks)
    for t in range(n_tasks):

        w_h = tikhonov_mean_solver(lmbd, h, X[t], y[t], verbose=verbose)

        # Testing
        mean_square_errors[t] = 0.5*(np.mean((X_test[t] @ w_h - y_test[t])**2))

        if verbose > 0:
            print('mse-test', mean_square_errors[t])

    return mean_square_errors


if __name__ == '__main__':
    from eperiments import exp1

    exp1()