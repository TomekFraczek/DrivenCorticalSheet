import json
import argparse
import sys

import numpy as np

from logger import Logger
from pathlib import Path
from modeling.model import KuramotoSystem, plot_interaction
from plotting.animate import Animator
from plotting.plot_solution import PlotSetup


CONFIG_NAME = 'config', 'json'
PHASES_NAME = 'oscillators', 'npy'
NOTES_NAME = 'notes', 'txt'
TIME_NAME = 'time', 'npy'
LOG_NAME = 'log', 'txt'


def model(config, fmt: (str, PlotSetup) = 'simulation'):
    """Prepare and run a cortical sheet simulation based on the passed config dictionary"""

    nodes_side = config['sqrt_nodes']
    time = config['time']
    gain_ratio = config['gain_ratio']

    # Init model
    gain = gain_ratio*nodes_side**2

    kuramoto = KuramotoSystem(
        (nodes_side, nodes_side),
        config['system'],
        gain,
        config['external_input'],
        config['external_input_weight']
    )

    # Run Model
    time_eval = np.linspace(0, time, config['frames'])

    try:
        solution = kuramoto.solve(
            (0, time),
            config['ODE_method'],  # too stiff for 'RK45', use 'LSODA',‘BDF’,'Radau'
            config['continuous_soln'],
            time_eval,
            config['max_delta_t'],
            config['zero_ics'],
        )
        osc_state = solution.y.reshape((nodes_side, nodes_side, solution.t.shape[0]))
        time = solution.t
        msg = solution.message
    except Exception as e:
        print('Integration failed!')
        osc_state = None
        time = None
        msg = f'{e}'

    if config['save_numpy']:
        fmt = save_data(fmt, config, osc_state, time, msg)

    return osc_state, time, fmt


def save_data(fmt, config, osc_state=None, time=None, solver_notes=None):
    """Load the result of a simulation run to the target directory"""
    import subprocess
    git_hash = subprocess.check_output(["git", "describe", "--output"]).strip()
    git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip()
    solver_notes += f'\n\nCode used for this run:\n' \
                    f'    git url: {git_url}\n' \
                    f'    commit hash: {git_hash}'

    fmt = fmt if hasattr(fmt, 'file_name') else PlotSetup(label=fmt)
    print(f'Saving to {fmt.directory}')

    with open(fmt.file_name(*CONFIG_NAME), 'w') as f:
        json.dump(config, f, indent=2)
    with open(fmt.file_name(*NOTES_NAME), 'w') as f:
        if solver_notes:
            f.write(solver_notes)
    if osc_state is not None:
        np.save(fmt.file_name(*PHASES_NAME), osc_state)
    if time is not None:
        np.save(fmt.file_name(*TIME_NAME), time)

    return fmt


def load_data(data_folder):
    """Load the data (result of a simulation run) from the target directory"""
    fmt = PlotSetup(base_folder=data_folder, readonly=True)
    with open(fmt.file_name(*CONFIG_NAME)) as f:
        config = json.load(f)
    osc_state = np.load(fmt.file_name(*PHASES_NAME))
    time = np.load(fmt.file_name(*TIME_NAME))

    return config, osc_state, time, fmt


def plot(config=None, osc_states=None, time=None, data_folder=None, fmt=None):
    """Plot a gif of the phase evolution over time for the given data, optionally loaded from disk"""

    # No data provided explicitly, need to load from the passed folder
    if data_folder is not None and (config is None or osc_states is None or time is None):
        config, osc_states, time, fmt = load_data(data_folder)

    # Insufficient info provided, we don't know what to plot!
    elif data_folder is None and (config is None or osc_states is None or time is None):
        raise KeyError('Both the data_folder and the data contents were left blank!')

    vid = Animator(config, fmt)
    vid.animate(osc_states, time, cleanup=False)


def run(config_set: str = 'local_sync', config_file: str = 'model_config.json', base_path='plots', do_plot=True):
    """Run a model based on the passed config file and immediately plot the results"""

    path_fmt = PlotSetup(base_path, config_set)
    sys.stdout = Logger(path_fmt.file_name(*LOG_NAME))

    with open(Path(config_file).resolve()) as f:
        var = json.load(f)
    config = var[config_set]

    oscillator_state, time, path_fmt = model(config, path_fmt)
    if do_plot:
        plot(config=config, osc_states=oscillator_state, time=time, fmt=path_fmt)
    plot_interaction(config['sqrt_nodes'], config['system'], config['gain_ratio'], out_fmt=path_fmt)


def main():
    """Function to run"""
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')

    parser.add_argument('--set', metavar='scenario | variable set',
                        type=str, nargs='?',
                        help='model_config.json key like global_sync',
                        default='test_set')

    parser.add_argument('--path', metavar='directory to config.json',
                        type=str, nargs='?',
                        help='model_config.json path, default is pwd',
                        default='model_config.json')

    parser.add_argument('--out', metavar='output directory',
                        type=str, nargs='?',
                        help='base path to output raw data and plots',
                        default='plots')

    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the results right away')

    args = parser.parse_args()
    run(args.set, args.path, do_plot=args.plot)


if __name__ == '__main__':
    main()
