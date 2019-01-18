import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os


def _plot_array(plot_f, metric, use_valid_str, alpha, x=None, label='online LTL', linestyle='-', color=None):
    y_mean_ltl = np.mean(metric, axis=-1)
    y_std_ltl = np.std(metric, axis=-1)

    plot_f(y_mean_ltl, label=label + use_valid_str, color=color, linestyle=linestyle)

    if x is None:
        x = range(len(y_mean_ltl))

    plt.fill_between(x=x, y1=y_mean_ltl + y_std_ltl / 2, y2=y_mean_ltl - y_std_ltl / 2, color=color, alpha=alpha)

    return x


def _plot_from_dict(plot_f, metric_dict, use_valid_str, alpha, x=None, label='online LTL', color='orange'):
    line_styles = [':', '-.', '--', '-']
    for is_name, metric in metric_dict.items():
        out = _plot_array(plot_f, metric, use_valid_str, alpha, x=x, label=label+'-'+is_name, linestyle=line_styles.pop(),
                color=color)
    return out


def _plot(plot_f, metric, use_valid_str, alpha, x=None, label='online LTL', color=None):
    if isinstance(metric, dict):
        return _plot_from_dict(plot_f, metric, use_valid_str, alpha, x, label=label, color=color)
    else:
        return _plot_array(plot_f, metric, use_valid_str, alpha, x, label=label, color=color)


def plot(metric_ltl, metric_itl, metric_oracle, metric_inner_initial=None, metric_inner_oracle=None, metric_wbar=None,
         use_valid_str='', y_label='', title='', save_dir_path=None, show_plot=True, filename='metric_test.png'):

    alpha = 0.1

    x = _plot(plt.plot, metric_ltl, use_valid_str, alpha, label='online LTL', color='orange')

    _plot(plt.axhline, metric_itl, '', alpha, x=x, label='ITL', color='red')
    _plot(plt.axhline, metric_oracle, '', alpha, x=x, label='Oracle', color='green')

    if metric_inner_initial is not None:
        y_mean_ii = np.mean(metric_inner_initial)
        y_std_ii = np.std(metric_inner_initial)

        plt.axhline(y_mean_ii, label='w = 0', color='blue', linestyle='--')
        plt.fill_between(x=x, y1=y_mean_ii + y_std_ii / 2, y2=y_mean_ii - y_std_ii / 2, color='blue',
                         alpha=alpha)

    if metric_inner_oracle is not None:
        y_mean_io = np.mean(metric_inner_oracle)
        y_std_io = np.std(metric_inner_oracle)

        plt.axhline(y_mean_io, label=r'$w = w_\mu$', color='purple', linestyle='--')
        plt.fill_between(x=x, y1=y_mean_io + y_std_io / 2, y2=y_mean_io - y_std_io / 2, color='purple',
                         alpha=alpha)

    if metric_wbar is not None:
        y_mean_wbar = np.mean(metric_wbar)
        y_std_wbar = np.std(metric_wbar)

        plt.axhline(y_mean_wbar, label=r'$w = \bar{w}$', color='green', linestyle='-.')
        plt.fill_between(x=x, y1=y_mean_wbar + y_std_wbar / 2, y2=y_mean_wbar - y_std_wbar / 2, color='green',
                         alpha=alpha)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('T')
    plt.xlim(right=x[-1], left=x[0])
    plt.legend()

    if save_dir_path is not None:
        pylab.savefig(os.path.join(save_dir_path, filename))
    if show_plot:
        plt.show()
    else:
        plt.close()
