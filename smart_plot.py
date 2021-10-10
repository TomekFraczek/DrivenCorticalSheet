import json
import os
import re
import argparse

import numpy as np
from matplotlib import pyplot as plt
from plotting.animate import animate_one
from plotting.common import completed_sims
from plotting.plotformat import PlotSetup
from plotting.fourier_space import fourier_1d, fourier_2d, plot_psd_width, plot_sweep_spread


# Collection of all the functions to plot on each individual directory
# Each entry should be the name of the primary output file as a key, with the function as the value
PLOT_FUNCTIONS = {
    'animation': animate_one,
    'fourier 1d': fourier_1d,
    'fourier 2d': fourier_2d
}
SWEEP_PLOTS = {
    'psd widths': plot_psd_width,
    'freq spreads': plot_sweep_spread
}


def plot_individual(source_dir):
    """Run all the single-run plotting functions currently listed"""
    for sim_folder in completed_sims(source_dir):
        for plot_name, plot_func in PLOT_FUNCTIONS.items():
            if not sim_folder.has_file(plot_name):
                plot_func(sim_folder)
                plt.close()


def plot_sweeps(source_dir):
    """Run all the sweep-wide plotting functions currently listed"""
    source_dir = PlotSetup(source_dir, build_new=False)
    for plot_name, plot_func in SWEEP_PLOTS.items():
        plot_func(source_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')
    parser.add_argument('--source', metavar='source directory',
                        type=str, nargs='?',
                        help='base path to source plot data from',
                        default='plots')

    args = parser.parse_args()
    plot_individual(args.source)
    plot_sweeps(args.source)


