import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os

#from matplotlib2tikz import save as tikz_save

std_mult = 3


def _plot_array(plot_f, metric, use_valid_str, alpha, x=None, label='online LTL', linestyle='-', color=None):
    if metric is None:
        return x

    y_mean_ltl = np.mean(metric, axis=-1)
    if len(y_mean_ltl.shape) == 0:
        y_mean_ltl = np.array([y_mean_ltl for _ in x])

    y_std_ltl = np.std(metric, axis=-1)

    if x is None:
        x = range(len(y_mean_ltl))

    plot_f(y_mean_ltl, label=label + use_valid_str, color=color, linestyle=linestyle)

    plt.fill_between(x=x, y1=y_mean_ltl + std_mult * y_std_ltl / 2, y2=y_mean_ltl - std_mult *y_std_ltl / 2, color=color, alpha=alpha)

    return x


def _plot_from_dict(plot_f, metric_dict, use_valid_str, alpha, x=None, label='online LTL', color='orange'):
    line_styles = [':', '-.', '--', '-']
    for is_name, metric in metric_dict.items():
        if is_name != '':
            li = label+' '+is_name
        else:
            li = label
        out = _plot_array(plot_f, metric, use_valid_str, alpha, x=x, label=li, linestyle=line_styles.pop(),
                color=color)
    return out


def _plot(metric, use_valid_str, alpha, x=None, label='online LTL', color=None, linestyle='-'):
    if isinstance(metric, dict):
        return _plot_from_dict(plt.plot, metric, use_valid_str, alpha, x, label=label, color=color)
    else:
        return _plot_array(plt.plot, metric, use_valid_str, alpha, x, label=label, color=color, linestyle=linestyle)


def plot(metric_ltl, metric_itl, metric_oracle, metric_inner_initial=None, metric_inner_oracle=None, metric_wbar=None,
         use_valid_str='', y_label='', title='', save_dir_path=None, show_plot=True, filename='metric_test.png'):

    alpha = 0.1

    x = _plot(metric_ltl, use_valid_str, alpha, label='online LTL', color='orange')
    _plot(metric_itl, '', alpha, x=x, label='ITL', color='red')
    _plot(metric_oracle, '', alpha, x=x, label='Oracle', color='green')
    _plot(metric_inner_initial, '', alpha, x=x, label='w = 0', color='blue', linestyle='--')
    _plot(metric_inner_oracle, '', alpha, x=x, label=r'$w = w_\mu$', color='purple', linestyle='--')
    _plot(metric_wbar, '', alpha, x=x, label=r'$w = \bar{w}$', color='green', linestyle='-.')

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('T')
    plt.xlim(right=x[-1], left=x[0])
    plt.legend()

    if save_dir_path is not None:
        pylab.savefig(os.path.join(save_dir_path, filename+'.png'))
        #pylab.savefig(os.path.join(save_dir_path, filename+'.pgf'))
        #tikz_save(os.path.join(save_dir_path, filename+'.txt'))
    if show_plot:
        plt.show()
    plt.close()


def plot_2fig(metric_ltl, metric_itl, metric_oracle, metric_inner_initial=None, metric_inner_oracle=None,
              metric_wbar=None, use_valid_str='', y_label='', title='', name='loss', save_dir_path=None, show_plot=True):
    plot(metric_ltl, metric_itl, metric_oracle, None, None, None,
         use_valid_str, y_label, title, save_dir_path, show_plot, name)

    plot(metric_ltl, metric_itl, metric_oracle, metric_inner_initial, metric_inner_oracle, metric_wbar,
         use_valid_str, y_label, title, save_dir_path, show_plot, name+'_plus')