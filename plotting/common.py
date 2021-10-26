import re
import json

import numpy as np

from plotting.plotformat import PlotSetup, from_existing

CONFIG_NAME = 'config', 'json'
PHASES_NAME = 'oscillators', 'npy'
NOTES_NAME = 'notes', 'txt'
TIME_NAME = 'time', 'npy'
LOG_NAME = 'log', 'txt'


def safe_search(string, pattern, group=1):
    try:
        match = re.search(pattern, string).group(group)
    except AttributeError:
        match = None
    return match


def get_point_id(sim_folder):
    return safe_search(sim_folder.directory, 'Point([0-9]*)')


def get_rep_id(sim_folder):
    return safe_search(sim_folder.directory, 'Rep([0-9]*)')


def count_reps(point_folder):
    reps = [f for f in point_folder.sub_folders() if get_rep_id(f)]
    return len(reps)


def completed_sims(source_dir):
    return [d for d in from_existing(source_dir) if d.has_file('completion', 'txt')]


def source_data(data_src, filename, calc_func, load=True):
    if load and data_src.has_file(filename):
        try:
            raw_data = np.load(data_src.file_name(filename, 'npy'), allow_pickle=False)
        except ValueError:
            raw_data = calc_func(data_src)
    else:
        raw_data = calc_func(data_src)

    return raw_data


def load_sim_time(data_src):
    return np.load(data_src.file_name(*TIME_NAME))


def load_config(data_src):
    try:
        f = open(data_src.file_name('config', 'json'))
    except FileNotFoundError:
        try:
            f = open(data_src.file_name('sweep_config', 'json'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find config.json or sweep_config.json in: \n   {data_src.directory}')
        else:
            conf = json.load(f)
            f.close()
    else:
        conf = json.load(f)
        f.close()

    return conf


def load_sim_results(data_folder):
    """Load the data (result of a simulation run) from the target directory"""
    if hasattr(data_folder, 'make_file_path'):
        fmt = data_folder
    else:
        fmt = PlotSetup(base_folder=data_folder, readonly=True)
    with open(fmt.file_name(*CONFIG_NAME)) as f:
        config = json.load(f)
    osc_state = np.load(fmt.file_name(*PHASES_NAME))
    time = load_sim_time(fmt)

    return config, osc_state, time, fmt


def calc_sweep(source_dir, function, save_name):
    """Calculate a function"""
    all_sims = completed_sims(source_dir)
    try:
        with open(all_sims[0].file_name('config', 'json')) as conf_file:
            point_conf = json.load(conf_file)['point_config']
    except IndexError:
        raise IndexError(f'No simulations available in: {source_dir}')
    x_vals = np.linspace(**point_conf['x-var'])
    y_vals = np.linspace(**point_conf['y-var'])
    n_xs, n_ys, = len(x_vals), len(y_vals)
    n_reps = count_reps(all_sims[0].parent())

    # Calculate all the desired values for each simulation, storing each value in a separate array
    all_values = []
    n_vals = None
    for i, sim_folder in enumerate(all_sims):
        values = function(sim_folder)

        # Identify where in the sweep we are, and save each of the calculated values
        point_id = get_point_id(sim_folder)
        rep_id = get_rep_id(sim_folder)
        if not all_values:
            n_vals = len(values) if hasattr(values, '__len__') else 1
            all_values = [np.zeros(shape=(n_xs * n_ys, n_reps)) for k in range(n_vals)]

        if n_vals > 1:
            for j, v in enumerate(values):
                all_values[j][int(point_id), int(rep_id)] = v
        else:
            all_values[0][int(point_id), int(rep_id)] = values

    # Average together the results at each point for each set of calculated values
    shaped_values = []
    for value_set in all_values:
        avg_values = np.mean(value_set, axis=1)
        shaped_values.append(np.reshape(avg_values, (n_xs, n_ys)))

    saveable = np.array([*shaped_values, *np.meshgrid(x_vals, y_vals)])
    np.save(source_dir.file_name(save_name, 'npy'), saveable, allow_pickle=False)
    return saveable


def calc_sweep_wrapper(function, save_name):
    """Wrapper to enable calc_sweep to be passed through source_data"""
    def sweep(source_dir):
        return calc_sweep(source_dir, function, save_name)
    return sweep


def match_varname(strings, variable):
    reg = f"\"([a-zA-Z_\-]+)\"[ ]?:[ ]?\"<{variable}-var>\""
    for s in strings:
        search = re.search(reg, s)
        if search:
            return search.group(1)
    else:
        raise KeyError(f'Failed to find the name of the {variable} variable')


def var_names(sweep_src):
    with open(sweep_src.file_name('config', 'json')) as f:
        lines = f.readlines()
    x_name = match_varname(lines, 'x')
    y_name = match_varname(lines, 'y')
    return x_name, y_name





