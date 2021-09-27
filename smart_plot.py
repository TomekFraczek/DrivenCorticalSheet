import json
import os
import re
import argparse

import numpy as np
from matplotlib import pyplot as plt
from plotting.plotformat import from_existing
from plotting.animate import animate_one
from plotting.fourier_space import fourier_1d, fourier_2d, psd_width


# Collection of all the functions to plot on each individual directory
# Each entry should be the name of the primary output file as a key, with the function as the value
PLOT_FUNCTIONS = {
    'animation': animate_one,
    'fourier 1d': fourier_1d,
    'fourier 2d': fourier_2d
}


def completed_sims(source_dir):
    return [d for d in from_existing(source_dir) if d.has_file('completion', 'txt')]


def plot_individual(source_dir):

    for sim_folder in completed_sims(source_dir):
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            if not sim_folder.has_file(plot_name):
                plot_func(sim_folder)
                plt.close()


def plot_sweep(source_dir):

    all_sims = completed_sims(source_dir)
    with open(all_sims[0].file_name('config', 'json')) as conf_file:
        point_conf = json.load(conf_file)['point_config']
    x_vals = np.linspace(**point_conf['x-var'])
    y_vals = np.linspace(**point_conf['y-var'])
    n_xs, n_ys, n_reps = len(x_vals), len(y_vals), point_conf['repetitions']

    all_vars = np.zeros(shape=(n_xs * n_ys, n_reps))
    all_means = np.zeros(shape=(n_xs * n_ys, n_reps))
    for i, sim_folder in enumerate(all_sims):
        point_id = re.search('Point([0-9]*)', sim_folder.directory).group(1)
        rep_id = re.search('Rep([0-9]*)', sim_folder.directory).group(1)
        psd_mean, psd_var = psd_width(sim_folder)
        all_vars[int(point_id), int(rep_id)] = psd_var
        all_means[int(point_id), int(rep_id)] = psd_mean

    avg_vars = np.mean(all_vars, axis=1)
    avg_means = np.mean(all_means, axis=1)
    shaped_vars = np.reshape(avg_vars, (n_xs, n_ys))
    shaped_means = np.reshape(avg_means, (n_xs, n_ys))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(*np.meshgrid(x_vals, y_vals), shaped_vars, shading='nearest')
    plt.xlabel('Gain Ratio')
    plt.ylabel('Beta')
    plt.title('Vars')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(*np.meshgrid(x_vals, y_vals), shaped_means, shading='nearest')
    plt.xlabel('Gain Ratio')
    plt.ylabel('Beta')
    plt.title('Means')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')
    parser.add_argument('--source', metavar='source directory',
                        type=str, nargs='?',
                        help='base path to source plot data from',
                        default='plots')

    args = parser.parse_args()
    plot_individual(args.source)
    plot_sweep(args.source)


