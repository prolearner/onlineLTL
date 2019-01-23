from experiments_ICML import lenk_multi_seed


for i in [16, 12, 10, 7, 5, 3, 2, 1]:
    lenk_multi_seed(n_train=i)
    lenk_multi_seed(reg=True, n_train=i)




