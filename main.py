"""
"""
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve())
if __name__ == '__main__' and __package__ is None:
    __package__ = 'kurosc'

from datetime import datetime as dt
import numpy as np
import json
import argparse

##
from model import KuramotoSystem
from lib.animate import Animate
from lib.plot_solution import plot_output, save_data

# annotate:
#
# note on mthds for ode soln
# https://scicomp.stackexchange.com/questions/27178/bdf-vs-implicit-runge-kutta-time-stepping


def run(config_set: str = 'local_sync', config_file: str = 'model_config.json'):

    with open(Path(config_file).resolve()) as f:
        var = json.load(f)
    config = var[config_set]

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

    osc_state = solution.y.reshape((nodes_side,
                                    nodes_side,
                                    solution.t.shape[0]
                                    ))

    print('\nsol.shape:', solution.y.shape,
          '\nt.shape:', solution.t.shape,
          '\nosc.shape:', osc_state.shape)

    # Data labeling
    param = lambda d: [''.join(f'{key}={value}') for (key,value) in d.items()]
    title = f'{nodes_side}_osc_with_kn={int(gain/nodes_side**2)}_at_t_{time}_'
    title += '_'.join(param(config['system']['interaction']))
    title += '_'+'_'.join(param(config['system']['kernel']))

    if save_numpy:
        print('\ndata save is set to:',save_numpy,'type', type(save_numpy),
        '\noutput to:', title, 'levels up from lib')
        save_data(solution,title)

    # Plotting & animation
    # kuramoto.plot_solution(osc_state[-1],solution.t[-1])

    plot_output(kuramoto, kuramoto.osc,
                osc_state, solution.t,
                config['inspect_t_samples'],
                config['inspect_t_seconds'],
                config['interpolate_plot'])

    vid = Animate(kuramoto.osc.plot_directory)
    vid.to_gif(None, config['frame_rate'], sort=True, clean=True)
    print(vid.img_name)

    #TODO post process numpy array to have time series or just hadle it in this chain


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
