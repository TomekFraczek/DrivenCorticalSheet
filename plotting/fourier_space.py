import os

import matplotlib.pyplot as plt
import numpy as np

from plotting.common import load_data, source_data, calc_sweep, calc_sweep_wrapper
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
    np.save(data_src.file_name('1d_fourier_data', 'npy'), saveable, allow_pickle=False)
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
    np.save(data_src.file_name('2d_fourier_data', 'npy'), saveable, allow_pickle=False)
    return state_psds, time, freqs


def dist_cov(x_vals, y_vals, freq_array):
    """Estimate the covariance matrix from two distributions, not sets of samples"""
    x_freqs = np.sum(freq_array, axis=0)
    y_freqs = np.sum(freq_array, axis=1)
    mean_x = np.average(x_vals, weights=x_freqs)
    mean_y = np.average(y_vals, weights=y_freqs)
    cov_xx = weighted_var(x_vals, x_freqs, average=mean_x)
    cov_yy = weighted_var(y_vals, y_freqs, average=mean_y)

    cov_xy = np.average(
        np.outer(x_vals-mean_x, y_vals-mean_y),
        weights=freq_array
    )
    return np.array([
        [cov_xx, cov_xy],
        [cov_xy, cov_yy]
    ])


def fourier_cov(data_src):

    state_ffts, time, freqs = source_fourier_2d(data_src)
    covariances = []
    for i in range(state_ffts.shape[0]):
        fft = state_ffts[i, :, :]
        cov_here = dist_cov(freqs, freqs, fft)
        covariances.append(cov_here)

    np.save(data_src.file_name('fourier_covariances', 'npy'), covariances, allow_pickle=False)
    return covariances


def source_fourier_2d(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, '2d_fourier_data', fourier_data_2d, load=load)
    state_ffts, time, freqs = raw_data[0, :, :], raw_data[1, :, :], raw_data[2, :, :]
    return state_ffts, time, freqs


def source_fourier_1d(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, '1d_fourier_data', fourier_data_2d, load=load)
    avg_psds, time, freqs = raw_data[0, :, :], raw_data[1, :, :], raw_data[2, :, :]
    return avg_psds, time, freqs


def source_fourier_cov(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, 'fourier_covariances', fourier_cov, load=load)
    return raw_data


def fourier_1d(data_src):
    avg_psds, time, freqs = source_fourier_1d(data_src)

    X, Y = np.meshgrid(time, freqs)

    fig = plt.figure()
    plt.pcolormesh(X, Y, avg_psds.T, shading='nearest')
    plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.savefig(data_src.file_name('fourier 1d', 'png'))


def fourier_2d(data_src):
    state_ffts, time, freqs = source_fourier_2d(data_src)

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


def collapsed_spread(data_src):
    spread = []
    covariances = source_fourier_cov(data_src)
    for i in range(covariances.shape[0]):
        cov = covariances[0, :, :]
        spread.append(
            np.sqrt(cov[0][0]*cov[1][1]-cov[0][1]**2)
        )
    np.save(data_src.file_name('2d_fourier_data', 'npy'), np.array(spread), allow_pickle=False)
    return spread


def weighted_var(values, weights, average=None):
    """Calculate the variance using a weighted mean"""
    average = np.average(values, weights=weights) if average is None else average
    variance = np.average((values - average) ** 2, weights=weights)
    return variance


def psd_width(data_src):

    # Ensure that the required data has been prepared
    if not data_src.has_file('1d_fourier_data', extension='npy'):
        fourier_1d(data_src)

    psd_data = np.load(data_src.file_name('1d_fourier_data', 'npy'), allow_pickle=True)
    times, freqs, psds = psd_data[0], psd_data[1], psd_data[2]

    end_psd = psds[-1, :]
    end_freqs = freqs[:, -1]

    average = np.average(end_freqs, weights=end_psd)
    variance = weighted_var(end_freqs, end_psd, average=average)

    return average, variance


def plot_psd_width(data_src):

    calc = calc_sweep_wrapper(psd_width, 'psd width')
    means, vars, xs, ys = source_data(data_src, 'psd width', calc)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(xs, ys, vars, shading='nearest')
    plt.xlabel('Gain Ratio')
    plt.ylabel('Beta')
    plt.title('Vars')
    plt.savefig(data_src.file_name('1D PSD Variances', 'png'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(xs, ys, means, shading='nearest')
    plt.xlabel('Gain Ratio')
    plt.ylabel('Beta')
    plt.title('Means')
    plt.savefig(data_src.file_name('1D PSD Averages', 'png'))
