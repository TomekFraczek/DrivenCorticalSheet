from matplotlib import pyplot as plt

from plotting.common import calc_sweep_wrapper, source_data
from plotting.common import load_sim_results, load_sim_time, var_names


def is_failed(data_source):

    with open(data_source.file_name('completion.txt')) as f:
        failure = 'failed' in f.read().lower()

    time = load_sim_time(data_source)
    return failure, time[-1]


def plot_failure_rates(data_source):

    calc = calc_sweep_wrapper(is_failed, 'failure_rates')
    failures, lasted, xs, ys = source_data(data_source, 'failure_rates', calc)

    x_name, y_name = var_names(data_source)

    fig = plt.figure(figsize=(11, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.pcolormesh(xs, ys, failures, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Failure Rate')

    ax = fig.add_subplot(1, 2, 2)
    ax.pcolormesh(xs, ys, lasted, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Simulated End Time')

    plt.tight_layout()
    plt.savefig(data_source.file_name('SweepFailureEvaluation', 'png'))

