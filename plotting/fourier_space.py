import matplotlib.pyplot as plt
import numpy as np

from plotting import load_data
from scipy.signal import welch


def fourier_data_1d(data_src):

    config, osc_states, time, _ = load_data(data_src)

    avg_psds = []
    for i, t in enumerate(time):
        state = osc_states[:, :, i]

        row_psds = []
        for row in state.T:
            freqs, pwrs = welch(row)
            row_psds.append(pwrs)

        row_psds = np.array(row_psds)
        avg_psd = np.mean(row_psds, axis=0)  # TODO: Check this is the right axis
        avg_psds.append(avg_psd)

    avg_psds = np.array(avg_psds)

    np.save(data_src.file_name('1d_fourier_data', 'npy'), avg_psds)
    return avg_psds, time, freqs


def fourier_1d(data_src):
    avg_psds, time, freqs = fourier_data_1d(data_src)

    X, Y = np.meshgrid(time, freqs)

    fig = plt.figure()
    plt.pcolormesh(X, Y, avg_psds.T)
    plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.savefig(data_src.file_name('fourier 1d', 'png'))

