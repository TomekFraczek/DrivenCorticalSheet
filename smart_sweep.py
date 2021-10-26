import os
import sys
import json
import random
import argparse
import traceback

from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from plotting.plotformat import PlotSetup, from_existing
from plotting.common import get_point_id, get_rep_id, load_config
from main import model, save_data, plot


MAX_TRIES = 5


def do_sweep(out_dir, config_name, n_jobs):

    path_fmt = PlotSetup(out_dir, config_name)

    with open('sweep_config.json') as f:
        sweep_config = json.load(f)[config_name]
    all_points = prep_points(path_fmt, sweep_config)

    sure_run(all_points, 0, n_jobs=n_jobs)


def redo_sweep(out_dir, n_jobs):
    """Restart a sweep that might have been ended prematurely"""
    existing = from_existing(out_dir)
    incomplete = check_complete(existing)
    sure_run(incomplete, 0, n_jobs=n_jobs)


def make_reps(point_path, rep_ids, config):
    reps = []
    for rep in rep_ids:
        rep_fmt = PlotSetup(point_path.directory, f'Rep{rep}')
        with open(rep_fmt.file_name('config', 'json'), 'w') as f:
            json.dump(config, f, indent=2)
        reps.append(rep_fmt)
    return reps


def more_reps(path_fmt, n_more_reps):
    """Add an additional round of repetitions to all points in the sweep"""
    new_runs = []

    path_fmt = PlotSetup(path_fmt, build_new=False)
    point_folders = [f for f in path_fmt.sub_folders() if get_point_id(f)]
    for point in point_folders:
        point_config = load_config(PlotSetup(point.directory, 'Rep0', build_new=False))

        old_max = 1 + max([int(get_rep_id(f)) for f in point.sub_folders() if get_rep_id(f)])
        new_max = old_max + n_more_reps

        new_runs.extend(
            make_reps(point, range(old_max, new_max), point_config)
        )
    return new_runs


def prep_points(path_fmt, sweep_config):
    """Prepare all folders and the configurations for each of the points in the sweep"""

    x_axis = np.linspace(**sweep_config['point_config']['x-var'])
    y_axis = np.linspace(**sweep_config['point_config']['y-var'])
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    with open(path_fmt.file_name('sweep_config', 'json'), 'w') as conf:
        json.dump(sweep_config, conf, indent=2)

    all_runs = []
    conf_str = json.dumps(sweep_config)
    for i, xy in enumerate(zip(x_mesh.flatten(), y_mesh.flatten())):
        x, y = xy[0], xy[1]
        conf_here = json.loads(conf_str.replace('"<x-var>"', str(x)).replace('"<y-var>"', str(y)))

        point_name = f'Point{i}'
        point_fmt = PlotSetup(path_fmt.directory, point_name)
        all_runs.extend(
            make_reps(point_fmt, range(conf_here["point_config"]["repetitions"]), conf_here)
        )
    return all_runs


def scramble(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest


def sure_run(point_fmts, tries, n_jobs=-1):
    """Run all points, recursively re-trying to run points that fail"""
    point_fmts = scramble(point_fmts)

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
    start_time = datetime.now()
    try:
        osc, time, fmt = model(config, point_fmt)
        err_message = None
    except Exception as e:
        osc, time, fmt = [], [], []
        err_message = ''.join(traceback.format_exception(*sys.exc_info()))
    finally:
        end_time = datetime.now()
        run_time = end_time - start_time

    # Check that the point has been run successfully
    if np.any(time) and time[-1] >= config['time']:
        with open(point_fmt.file_name('completion', 'txt'), 'w') as f:
            f.write('Simulation Completed\n\n')
            f.write(f'Run Time: {run_time}')
    else:
        error_msg = err_message if err_message else 'Error Unknown!'
        with open(point_fmt.file_name('completion', 'txt'), 'w') as f:
            f.write(f'Simulation failed to complete!\n    {error_msg}\n\n')
            f.write(f'Run Time: {run_time}')


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

    parser.add_argument('--restart', action='store_true',
                        help='Whether to try and restart a previously started run')

    parser.add_argument('--addreps', metavar='add more reps',
                        type=int, nargs='?',
                        help='The number of additional repetitions to run at each point',
                        default=0)

    args = parser.parse_args()

    if args.addreps:
        more_reps(args.out, args.addreps)
        args.restart = True

    if args.restart:
        redo_sweep(args.out, args.jobs)
    else:
        do_sweep(args.out, args.config, args.jobs)



if __name__ == '__main__':
    main()
