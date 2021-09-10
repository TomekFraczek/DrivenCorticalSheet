import os
import json
import argparse

import numpy as np
from joblib import Parallel, delayed

from plotting.plotformat import PlotSetup
from main import model, save_data, plot


MAX_TRIES = 5


def do_sweep(out_dir, config_name):

    path_fmt = PlotSetup(out_dir, config_name)

    with open('sweep_configs.json') as f:
        sweep_config = json.load(f)[config_name]
    all_points = prep_points(out_dir, sweep_config)

    sure_run(path_fmt, all_points, 0)


def prep_points(out_dir, sweep_config):
    """Prepare all folders and the configurations for each of the points in the sweep"""
    point_config = json.dumps(sweep_config['point_config'])

    x_axis = np.linspace(**sweep_config['x-var'])
    y_axis = np.linspace(**sweep_config['y-var'])
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    points = []
    for i, (x, y) in enumerate(zip(np.flatten(x_mesh), np.flatten(y_mesh))):
        conf_here = json.loads(point_config.replace('<x-var>', x).replace('<y-var>', y))

        point_name =  f'Point {i}'
        point_dir = os.path.join(out_dir, point_name)
        os.mkdir(point_dir)
        with open(os.path.join(point_dir, 'config.json'), 'w') as f:
            json.dump(conf_here, f)
        points.append(point_name)
    return points


def sure_run(path_fmt, point_list, tries, n_jobs=-1):
    """Run all points, recursively re-trying to run points that fail"""
    # Send command to run all points in parallel
    point_paths = [os.path.join(path_fmt, point_list)]
    Parallel(n_jobs=n_jobs, verbose=20)(
        delayed(model)(p, path_fmt) for p in point_paths
    )

    # Check which points failed to finish
    incomplete = check_complete(path_fmt)

    # Recursively run all incomplete points
    if tries <= MAX_TRIES and incomplete:
        tries += 1
        sure_run(path_fmt, incomplete, tries)


def run_point(point_path, fmt):
    # This function should be run in parallel

    # Load config from file
    with open(os.path.join(point_path, 'config.json')) as f:
        config = json.load(f)

    # Run the simulation at this point and save the data
    osc, time, fmt = model(config, fmt)

    # Check that the point has been run successfully
    if time[-1] >= config['time']:
        with open(fmt.file_name('completion', 'txt')) as f:
            f.write('Simulation Completed')
    else:
        error_msg = 'Error unknown!'
        with open(fmt.file_name('completion', 'txt')) as f:
            f.write(f'Simulation failed to complete!\n    {error_msg}')


def check_complete(path_fmt):

    folders_here = [f for f in path_fmt.directory.iterdir() if f.is_dir()]
    incomplete = [f for f in folders_here if not os.path.exists(os.path.join(f, 'completion.txt'))]
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

    args = parser.parse_args()
    do_sweep(args.out, args.config)


if __name__ == '__main__':
    main()
