import os

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

    n_osc = osc_states[:, :, 0].shape[-1]
    half = int(n_osc/2)

    state_psds = []
    for i, t in enumerate(time):
        state = osc_states[:, :, i]

        state_fft = fft2(state)
        state_fft[0, 0] = 0
        full_state_psd = np.real(state_fft * np.conj(state_fft))
        state_psd = full_state_psd[:half, :half]
        state_psds.append(state_psd)

    state_psds = np.array(state_psds)
    raw_freqs = fftfreq(n_osc)
    freqs = raw_freqs[:half]

    saveable = np.array([state_psds, np.meshgrid(freqs, freqs)])
    np.save(data_src.file_name('2d_fourier_data', 'npy'), saveable)
    return state_psds, time, freqs


def fourier_1d(data_src):
    avg_psds, time, freqs = fourier_data_1d(data_src)

    X, Y = np.meshgrid(time, freqs)

    fig = plt.figure()
    plt.pcolormesh(X, Y, avg_psds.T, shading='nearest')
    plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.savefig(data_src.file_name('fourier 1d', 'png'))


def fourier_2d(data_src):
    state_ffts, time, freqs = fourier_data_2d(data_src)

    fx, fy = np.meshgrid(freqs, freqs)

    fig = plt.figure(figsize=(20, 13))
    n_states = 7
    for i in range(n_states-1):
        plt.subplot(2, 3, i+1, aspect=1.0)
        state_id = round((i / n_states) * len(time))
        next_state = round((i+1 / n_states) * len(time))
        these_states = state_ffts[state_id:next_state, :, :]
        avg_state = np.mean(these_states, axis=0)
        plt.pcolormesh(fx, fy, avg_state, shading='nearest')
        plt.xlabel('X Frequencies')
        plt.ylabel('Y Frequencies')
        plt.colorbar()
        plt.title(f'T = {round(time[state_id], 2)}')

    plt.tight_layout()
    plt.savefig(data_src.file_name('fourier 2d', 'png'))


def psd_width(data_src):

    # Ensure that the required data has been prepared
    if not data_src.has_file('1d_fourier_data', extension='npy'):
        fourier_1d(data_src)

    psd_data = np.load(data_src.file_name('1d_fourier_data', 'npy'), allow_pickle=True)
    times, freqs, psds = psd_data[0], psd_data[1], psd_data[2]

    end_psd = psds[-1, :]
    end_freqs = freqs[:, -1]

    average = np.average(end_freqs, weights=end_psd)
    variance = np.average((end_freqs - average) ** 2, weights=end_psd)

    return average, variance

