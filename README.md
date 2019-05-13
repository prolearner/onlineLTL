# onlineLTL
python implementation of online learning to learn non-smooth algorithms.

## reproducting experiments from "Learning-to-Learn Stochastic Gradient Descent with Biased Regularization (https://arxiv.org/abs/1903.10399v1)"

For the synthetic experiments run `exp_synthetic.py` while for the computer survey experiments run `exp_lenk.py`.

You can find the implementation of the algorithms discussed in the paper inside `algorithms.py`, while the dataset generation
and loading functions are in `data/data_generator.py` and `data/data_load.py`

Experiments results will be stored in a folder inside `exps` with a descriptive name containing details about the
experiments' parameters (more details in `experiments.py` and `train.py`)

If you have any problems feel free to contact me or open an issue.


