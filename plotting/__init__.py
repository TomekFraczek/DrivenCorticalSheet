import json
import numpy as np
from plotting.plotformat import PlotSetup


CONFIG_NAME = 'config', 'json'
PHASES_NAME = 'oscillators', 'npy'
NOTES_NAME = 'notes', 'txt'
TIME_NAME = 'time', 'npy'
LOG_NAME = 'log', 'txt'


def load_data(data_folder):
    """Load the data (result of a simulation run) from the target directory"""
    if hasattr(data_folder, 'make_file_path'):
        fmt = data_folder
    else:
        fmt = PlotSetup(base_folder=data_folder, readonly=True)
    with open(fmt.file_name(*CONFIG_NAME)) as f:
        config = json.load(f)
    osc_state = np.load(fmt.file_name(*PHASES_NAME))
    time = np.load(fmt.file_name(*TIME_NAME))

    return config, osc_state, time, fmt