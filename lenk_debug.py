from data import data_load
import algorithms as algs
import losses
import numpy as np
from scipy import io as sio

seed = 3
cla = False
np.random.seed(seed)


temp = sio.loadmat('data/lenk_data.mat')
train_data = temp['Traindata']
test_data = temp['Testdata']

Y = train_data[:, 14]
Y_test = test_data[:, 14]


def threshold_for_classifcation(Y, th):
    Y_bc = np.ones_like(Y)
    Y_bc[Y < th] = -1
    return Y_bc


if cla:
    Y = threshold_for_classifcation(Y, 5)
    Y_test = threshold_for_classifcation(Y_test, 5)

X = train_data[:, :14]
X_test = test_data[:, :14]

n_tasks = 180
ne_tr = 16   # numer of elements on train set per task
ne_test = 4  # numer of elements on test set per task


def split_tasks(data, nt, number_of_elements):
    return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(nt)]


data_m = split_tasks(X, n_tasks, ne_tr)
labels_m = split_tasks(Y, n_tasks, ne_tr)

data_test_m = split_tasks(X_test, n_tasks, ne_test)
labels_test_m = split_tasks(Y_test, n_tasks, ne_test)

task_range_tr = np.random.permutation(n_tasks)

# data_train, data_val, data_test = data_load.computer_data_ge_reg(n_train_tasks=180, n_val_task=0)

# ITL
solvers = [algs.InnerSubGD, algs.FISTA]
losses_train = []
n_train = 8  # 8 as in argyriou et al. 2007 (pag 6)
n_iter = 200
for solver_class in solvers:
    for t in [80]:
        X_train, Y_train = data_m[t][:n_train], labels_m[t][:n_train]
        X_test, Y_test = data_test_m[t], labels_test_m[t]

        solver = solver_class(lmbd=0.46415888336127775, h=np.zeros_like(X_train[0]), loss_class=losses.AbsoluteLoss)
        solver(X_train, Y_train, verbose=0, n_iter=n_iter)

        losses_train.append(solver.train_loss(X_train, Y_train, n_iter-1))
        #print('train-loss ' + str(t), losses_train[t])

print('train-losses', losses_train)
print(np.sqrt(2))




