from experiments import school_multi_seed


def exp1_school():
    school_multi_seed(seeds=[0], inner_solver_test_str=['ssubgd'], inner_solver_str=['ssubgd'], n_tasks=2000,
                      alphas=[10], lambdas=[0.1])


# exp1_school()
# school_multi_seed()

exp1_school()