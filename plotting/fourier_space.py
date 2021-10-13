import os

import matplotlib.pyplot as plt
import numpy as np

from plotting.common import load_sim_results, source_data, calc_sweep_wrapper, var_names, load_sim_time
from scipy.signal import welch
from scipy.fft import fft2, fftfreq, fftshift


def fourier_data_1d(data_src):

    config, osc_states, time, _ = load_sim_results(data_src)

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

    avg_psds = np.array(avg_psds).T

    saveable = np.array([avg_psds, *np.meshgrid(time, freqs)])
    np.save(data_src.file_name('1d_fourier_data', 'npy'), saveable, allow_pickle=False)
    return saveable


def fourier_data_2d(data_src):
    config, osc_states, time, _ = load_sim_results(data_src)

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

    saveable = np.array([*state_psds, *np.meshgrid(freqs, freqs)])
    np.save(data_src.file_name('2d_fourier_data', 'npy'), saveable, allow_pickle=False)
    return saveable


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

    state_ffts, x_freqs, _ = source_fourier_2d(data_src)
    freqs = np.unique(x_freqs)
    covariances = []
    for i in range(state_ffts.shape[0]):
        fft = state_ffts[i, :, :]
        cov_here = dist_cov(freqs, freqs, fft)
        covariances.append(cov_here)

    covariances = np.array(covariances)
    np.save(data_src.file_name('fourier_covariances', 'npy'), covariances, allow_pickle=False)
    return covariances


def means_2d(data_src):
    state_ffts, x_weight, _ = source_fourier_2d(data_src)
    freqs = np.unique(x_weight)
    means = []
    for i in range(state_ffts.shape[0]):
        fft = state_ffts[i, :, :]
        x_weight = np.sum(fft, axis=0)
        y_weight = np.sum(fft, axis=1)
        mean_x = np.average(freqs, weights=x_weight)
        mean_y = np.average(freqs, weights=y_weight)
        means.append([mean_x, mean_y])

    covariances = np.array(means)
    np.save(data_src.file_name('fourier_2d_means', 'npy'), covariances, allow_pickle=False)
    return covariances


def source_fourier_2d(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, '2d_fourier_data', fourier_data_2d, load=load)
    state_ffts, x_freqs, y_freqs = raw_data[:-2, :, :], raw_data[-2, :, :], raw_data[-1, :, :]
    return state_ffts, x_freqs, y_freqs


def source_fourier_1d(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, '1d_fourier_data', fourier_data_1d, load=load)
    avg_psds, time, freqs = raw_data[:-2, :, :], raw_data[-2, :, :], raw_data[-1, :, :]
    return avg_psds, time, freqs


def source_fourier_cov(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, 'fourier_covariances', fourier_cov, load=load)
    return raw_data


def source_fourier_means(data_src, load=True):
    """Get 2d fourier transform data, loading from existing file if possible"""
    raw_data = source_data(data_src, 'fourier_2d_means', means_2d, load=load)
    return raw_data


def source_spread(data_src, load=True):
    """"""
    spreads = source_data(data_src, 'collapsed_spread', collapsed_spread, load=load)
    return spreads


def fourier_1d(data_src):
    avg_psds, freqs, time = source_fourier_1d(data_src)

    fig = plt.figure()
    plt.pcolormesh(freqs, time, avg_psds[0], shading='nearest')
    plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.savefig(data_src.file_name('fourier 1d', 'png'))


def fourier_2d(data_src):
    state_ffts, fx, fy = source_fourier_2d(data_src)
    time = load_sim_time(data_src)

    fig = plt.figure(figsize=(20, 13))
    n_states = 6
    for i in range(n_states):
        plt.subplot(2, 3, i+1, aspect=1.0)
        state_id = round((i / (n_states-1)) * len(time))
        try:
            this_state = state_ffts[state_id, :, :]
        except IndexError:
            state_id = -1
            this_state = state_ffts[-1, :, :]
        plt.pcolormesh(fx, fy, this_state, shading='nearest')
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
    np.save(data_src.file_name('collapsed_spread', 'npy'), np.array(spread), allow_pickle=False)
    return spread


def end_spread(data_src):
    spreads = source_spread(data_src)
    return spreads[-1]


def end_xy_vars(data_src):
    """Return the x and y covariances at this point"""
    covs = source_fourier_cov(data_src)
    end = covs[-1, :, :]
    return end[0, 0], end[1, 1]


def end_xy_means(data_src):
    """Return the x and y mean frequency at the end of this one run"""
    means = source_fourier_means(data_src)
    return means[-1]


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
    psds, times, freqs = psd_data[0], psd_data[1], psd_data[2]

    end_psd = psds[:, -1]
    end_freqs = freqs[:, -1]

    average = np.average(end_freqs, weights=end_psd)
    variance = weighted_var(end_freqs, end_psd, average=average)

    return average, variance


def plot_means_2d(data_src):
    """Plot the evolution of 2d fourier space means over time for a single run"""
    raw_means = source_fourier_means(data_src)
    time = load_sim_time(data_src)
    x_means, y_means = raw_means[:, 0], raw_means[:, 1]

    fig = plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(time, x_means)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean X Freq')

    plt.subplot(2, 1, 2)
    plt.plot(time, y_means)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Y Freq')

    plt.tight_layout()
    plt.savefig(data_src.file_name('FourierMeanEvolution'))
    plt.close()


def plot_vars_2d(data_src):
    """Plot the evolution of 2d fourier-space variances over time for a single run"""
    raw_covs = source_fourier_cov(data_src)
    time = load_sim_time(data_src)
    x_vars, y_vars = [], []
    for c in raw_covs:
        x_vars.append(c[0][0])
        y_vars.append(c[1][1])

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(time, x_vars)
    plt.xlabel('Time (s)')
    plt.ylabel('X Freq Variance')

    plt.subplot(2, 1, 2)
    plt.plot(time, y_vars)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Freq Variance')

    plt.tight_layout()
    plt.savefig(data_src.file_name('FourierVarianceEvolution'))
    plt.close()


def plot_sweep_spread(data_src):
    calc = calc_sweep_wrapper(end_spread, 'freq spreads')
    spreads, xs, ys = source_data(data_src, 'freq spreads', calc)
    x_name, y_name = var_names(data_src)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(xs, ys, spreads, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency Spreads')
    plt.savefig(data_src.file_name('Sweep Spreads', 'png'))


def plot_psd_width(data_src):

    calc = calc_sweep_wrapper(psd_width, 'psd width')
    means, vars, xs, ys = source_data(data_src, 'psd width', calc)
    x_name, y_name = var_names(data_src)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(xs, ys, vars, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency Variances')
    plt.savefig(data_src.file_name('1D PSD Variances', 'png'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    ax.pcolormesh(xs, ys, means, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency Averages')
    plt.savefig(data_src.file_name('1D PSD Averages', 'png'))


def plot_end_xy_vars(data_src):

    calc = calc_sweep_wrapper(end_xy_vars, 'endxy vars')
    x_vars, y_vars, xs, ys = source_data(data_src, 'end xy vars', calc)
    x_name, y_name = var_names(data_src)

    fig = plt.figure(figsize=(11, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.pcolormesh(xs, ys, x_vars, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency x-variance')

    ax = fig.add_subplot(1, 2, 2)
    ax.pcolormesh(xs, ys, y_vars, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency y-variance')

    plt.tight_layout()
    plt.savefig(data_src.file_name('2D PSD Variances', 'png'))


def plot_sweep_end_means(data_src):
    """Plot the evolution of 2d means over time for a single run"""
    calc = calc_sweep_wrapper(end_xy_means, 'end xy means')
    x_means, y_means, xs, ys = source_data(data_src, 'end xy means', calc)
    x_name, y_name = var_names(data_src)

    fig = plt.figure(figsize=(11, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.pcolormesh(xs, ys, x_means, shading='nearest')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Ending Frequency x-means')

    ax = fig.add_subplot(1, 2, 2)
    ax.pcolormesh(xs, ys, y_means, shading='nearest')
    plt.xlabel(x_name)
    plt.title('Ending Frequency y-means')

    plt.tight_layout()
    plt.savefig(data_src.file_name('2D PSD Means', 'png'))
