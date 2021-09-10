import os
import sys
import json
import argparse
import traceback

import numpy as np
from joblib import Parallel, delayed

from plotting.plotformat import PlotSetup
from main import model, save_data, plot


MAX_TRIES = 5


def do_sweep(out_dir, config_name, n_jobs):

    path_fmt = PlotSetup(out_dir, config_name)

    with open('sweep_config.json') as f:
        sweep_config = json.load(f)[config_name]
    all_points = prep_points(path_fmt, sweep_config)

    sure_run(all_points, 0, n_jobs=n_jobs)


def prep_points(path_fmt, sweep_config):
    """Prepare all folders and the configurations for each of the points in the sweep"""

    x_axis = np.linspace(**sweep_config['point_config']['x-var'])
    y_axis = np.linspace(**sweep_config['point_config']['y-var'])
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    points = []
    conf_str = json.dumps(sweep_config)
    for i, xy in enumerate(zip(x_mesh.flatten(), y_mesh.flatten())):
        x, y = xy[0], xy[1]
        conf_here = json.loads(conf_str.replace('"<x-var>"', str(x)).replace('"<y-var>"', str(y)))

        point_name = f'Point {i}'
        point_fmt = PlotSetup(path_fmt.directory, point_name)
        with open(point_fmt.file_name('config', 'json'), 'w') as f:
            json.dump(conf_here, f, indent=2)
        points.append(point_fmt)
    return points


def sure_run(point_fmts, tries, n_jobs=-1):
    """Run all points, recursively re-trying to run points that fail"""
    # Send command to run all points in parallel
    Parallel(n_jobs=n_jobs, verbose=20)(
        delayed(run_point)(p) for p in point_fmts
    )

    # Check which points failed to finish
    incomplete = check_complete(point_fmts)

    # Recursively run all incomplete points
    if tries <= MAX_TRIES and incomplete:
        tries += 1
        sure_run(point_fmts, incomplete, tries)


def run_point(point_fmt):
    # This function should be run in parallel

    # Load config from file
    with open(point_fmt.file_name('config', 'json')) as f:
        config = json.load(f)

    # Run the simulation at this point and save the data
    try:
        osc, time, fmt = model(config, point_fmt)
        err_message = None
    except Exception as e:
        osc, time, fmt = [], [], []
        err_message = ''.join(traceback.format_exception(*sys.exc_info()))

    # Check that the point has been run successfully
    if np.any(time) and time[-1] >= config['time']:
        with open(point_fmt.file_name('completion', 'txt'), 'w') as f:
            f.write('Simulation Completed')
    else:
        error_msg = err_message if err_message else 'Error Unknown!'
        with open(point_fmt.file_name('completion', 'txt'), 'w') as f:
            f.write(f'Simulation failed to complete!\n    {error_msg}')


def check_complete(point_fmts):

    incomplete = [fmt for fmt in point_fmts if not os.path.exists(fmt.file_name('completion', 'txt'))]
    return incomplete


def main():
    """Function to run the sweep from the commandline"""
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')

    parser.add_argument('--out', metavar='output directory',
                        type=str, nargs='?',
                        help='base path to output raw data and plots',
                        default='plots')

    parser.add_argument('--config', metavar='sweep configuration',
                        type=str, nargs='?',
                        help='name of sweep configuration to load from sweep_config.json',
                        default='plots')

    parser.add_argument('--jobs', metavar='num jobs',
                        type=int, nargs='?',
                        help='Number of parallel jobs to run',
                        default=-1)

    args = parser.parse_args()
    do_sweep(args.out, args.config, args.jobs)


if __name__ == '__main__':
    main()
