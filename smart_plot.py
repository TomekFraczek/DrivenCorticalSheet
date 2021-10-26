import json
import os
import re
import argparse

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from plotting.animate import animate_one
from plotting.common import completed_sims
from plotting.plotformat import PlotSetup
from plotting.meta_plottin import plot_failure_rates, plot_interaction_wrapped
from plotting.spatial_metrics import plot_sync_time, mega_gif, plot_stim_strength
from plotting.fourier_space import fourier_1d, fourier_2d, plot_psd_width, plot_sweep_spread, plot_end_xy_vars, \
    plot_sweep_end_means, collapsed_spread, plot_means_2d, plot_vars_2d


# Collection of all the functions to plot on each individual directory
# Each entry should be the name of the primary output file as a key, with the function as the value
PLOT_FUNCTIONS = {
    'animation': animate_one,
    'interaction': plot_interaction_wrapped,
    'fourier 1d': fourier_1d,
    'fourier 2d': fourier_2d,
    'collapsed_spread': collapsed_spread,
    'RelativeStimStrength': plot_stim_strength,
    'SynchronizationEvolution': plot_sync_time,
    'FourierMeanEvolution': plot_means_2d,
    'FourierVarianceEvolution': plot_vars_2d
}
SWEEP_PLOTS = {
    'psd widths': plot_psd_width,
    'freq spreads': plot_sweep_spread,
    'end xy vars': plot_end_xy_vars,
    'end xy means': plot_sweep_end_means,
    'mega gif': mega_gif
}


def plot_folder(sim_folder):
    print(f'Processing {sim_folder}..')
    for plot_name, plot_func in PLOT_FUNCTIONS.items():
        if not sim_folder.has_file(plot_name):
            print(f'    Running {plot_name}...')
            plot_func(sim_folder)
            plt.close()


def plot_individual(source_dir, n_jobs=-1):
    """Run all the single-run plotting functions currently listed"""
    Parallel(n_jobs=n_jobs, verbose=20)(
        delayed(plot_folder)(sim_folder)
        for sim_folder in completed_sims(source_dir)
    )


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
    plot_failure_rates(PlotSetup(args.source, build_new=False))
    plot_individual(args.source)
    plot_sweeps(args.source)


