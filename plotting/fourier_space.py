import matplotlib.pyplot as plt
import numpy as np

from plotting import load_data
from scipy.signal import welch
from scipy.fft import fft2, fftfreq, fftshift


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

    saveable = np.array([avg_psds, *np.meshgrid(time, freqs)])
    np.save(data_src.file_name('1d_fourier_data', 'npy'), saveable)
    return avg_psds, time, freqs


def fourier_data_2d(data_src):
    config, osc_states, time, _ = load_data(data_src)

    state_psds = []
    for i, t in enumerate(time):
        state = osc_states[:, :, i]

        state_fft = fft2(state)
        state_psd = fftshift(np.real(state_fft * np.conj(state_fft)))
        state_psds.append(state_psd)

    state_psds = np.array(state_psds)
    n_osc = state.shape[-1]
    raw_freqs = fftfreq(n_osc)
    freqs = np.array([*raw_freqs[round(n_osc/2):], *raw_freqs[:round(n_osc/2)]])

    saveable = np.array([state_psds, np.meshgrid(freqs, freqs)])
    np.save(data_src.file_name('2d_fourier_data', 'npy'), saveable)
    return state_psds, time, freqs


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


def fourier_2d(data_src):
    state_ffts, time, freqs = fourier_data_2d(data_src)

    fx, fy = np.meshgrid(freqs, freqs)

    fig = plt.figure(figsize=(20, 13))
    n_states = 6
    for i in range(n_states):
        plt.subplot(2, 3, i+1, aspect=1.0)
        state_id = round((i / n_states) * len(time))
        plt.pcolormesh(fx, fy, state_ffts[state_id, :, :])
        plt.xlabel('X Frequencies')
        plt.ylabel('Y Frequencies')
        plt.colorbar()
        plt.title(f'T = {round(time[state_id], 2)}')

    plt.tight_layout()
    plt.savefig(data_src.file_name('fourier 2d', 'png'))