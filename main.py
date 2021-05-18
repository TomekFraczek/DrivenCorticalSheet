import os
import json
import argparse

import numpy as np

from pathlib import Path
from modeling.model import KuramotoSystem
from plotting.animate import Animator
from plotting.plot_solution import plot_output, PlotSetup


CONFIG_NAME = 'config', 'json'
PHASES_NAME = 'oscillators', 'npy'
TIME_NAME = 'time', 'npy'


def model(config, label='simulation'):
    # Load all variables from specified set in json
    nodes_side = config['sqrt_nodes']
    time = config['time']

    save_numpy = config['save_numpy']

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

    solution = kuramoto.solve(
        (0, time),
        config['ODE_method'],  # too stiff for 'RK45', use 'LSODA',‘BDF’,'Radau'
        config['continuous_soln'],
        time_eval,
        config['max_delta_t'],
        config['zero_ics'],
    )

    osc_state = solution.y.reshape((nodes_side, nodes_side, solution.t.shape[0]))

    # Data labeling
    param = lambda d: [''.join(f'{key}={value}') for (key, value) in d.items()]
    title = f'{nodes_side}_osc_with_kn={int(gain/nodes_side**2)}_at_t_{time}_'
    title += '_'.join(param(config['system']['interaction']))
    title += '_'+'_'.join(param(config['system']['kernel']))

    if save_numpy:
        label = save_data(label, config, osc_state, solution.t)

    return osc_state, solution.t, label


def save_data(label, config, osc_state, time):
    fmt = PlotSetup(label=label)
    with open(fmt.file_name(*CONFIG_NAME), 'w') as f:
        json.dump(config, f, indent=2)
    np.save(fmt.file_name(*PHASES_NAME), osc_state)
    np.save(fmt.file_name(*TIME_NAME), time)

    return fmt.label


def load_data(data_folder):
    fmt = PlotSetup(base_folder=data_folder, readonly=True)
    with open(fmt.file_name(*CONFIG_NAME)) as f:
        config = json.load(f)
    osc_state = np.load(fmt.file_name(*PHASES_NAME))
    time = np.load(fmt.file_name(*TIME_NAME))

    return config, osc_state, time, fmt


def plot(config=None, osc_states=None, time=None, data_folder=None, label=None):

    # No data provided explicitly, need to load from the passed folder
    if data_folder is not None and (config is None or osc_states is None or time is None):
        config, osc_states, time, fmt = load_data(data_folder)

    # Insufficient info provided, we don't know what to plot!
    elif data_folder is None and (config is None or osc_states is None or time is None):
        raise KeyError('Both the data_folder and the data contents were left blank!')

    else:
        fmt = PlotSetup(label=label, readonly=True)

    vid = Animator(config, fmt)
    vid.animate(osc_states, time, cleanup=False)


def run(config_set: str = 'local_sync', config_file: str = 'model_config.json'):

    with open(Path(config_file).resolve()) as f:
        var = json.load(f)
    config = var[config_set]

    oscillator_state, time, label = model(config, label=config_set)
    plot(config=config, osc_states=oscillator_state, time=time, label=label)


def main():
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')

    parser.add_argument('--set', metavar='scenario | variable set',
                        type=str, nargs='?',
                        help='model_config.json key like global_sync',
                        default='test_set')

    parser.add_argument('--path', metavar='directory to config.json',
                        type=str, nargs='?',
                        help='model_config.json path, default is pwd',
                        default='model_config.json')

    args = parser.parse_args()
    run(args.set, args.path)



if __name__ == '__main__':
    main()
