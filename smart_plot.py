import os
import argparse

from matplotlib import pyplot as plt
from plotting.plotformat import from_existing
from plotting.animate import animate_one
from plotting.fourier_space import fourier_1d, fourier_2d


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')
    parser.add_argument('--source', metavar='source directory',
                        type=str, nargs='?',
                        help='base path to source plot data from',
                        default='plots')

    args = parser.parse_args()
    plot_individual(args.source)


