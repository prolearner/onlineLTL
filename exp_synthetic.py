from experiments_ICML import exp_multi_seed, exp_meta_val


def exp_class():
    for n_train in [10, 50, 100]:
        for tasks_std in [0.5, 1, 2, 3, 4]:
            exp_multi_seed('exp2', n_train=n_train, n_tasks=1000, w_bar=4, y_snr=1, task_std=tasks_std,
                            use_hyper_bounds=True, inner_solver_str=['ssubgd'], search_oracle=False)


def exp_reg():
    for n_train in [10, 50, 100]:
        for y_snr in [0.5, 1, 2]:
            for tasks_std in [0.5, 1, 2, 4]:
                    exp_multi_seed('exp1', n_train=n_train, n_tasks=1000, w_bar=4, y_snr=y_snr,
                                   task_std=tasks_std,
                                   use_hyper_bounds=True, inner_solver_str=['ssubgd'])


#exp_reg()
#exp_class()
#exp_meta_val('exp2', n_train=10, n_tasks=600, w_bar=4, y_snr=1, task_std=1, lambdas=1, alphas=1, use_hyper_bounds=True, inner_solver_str=['ssubgd'])

exp_multi_seed('exp2', n_train=10, n_tasks=1000, w_bar=4, y_snr=1, task_std=1, use_hyper_bounds=True, inner_solver_str=['ssubgd'])

