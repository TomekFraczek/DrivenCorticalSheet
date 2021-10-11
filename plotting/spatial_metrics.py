import os
import numpy as np

from plotting.common import load_sim_results, source_data, load_sim_time
from matplotlib import pyplot as plt


def state_sync(osc_phases):
    all_phases = osc_phases.flatten()
    imag_sync = np.sum(np.exp(1j * all_phases))/len(all_phases)
    return np.abs(imag_sync), np.angle(imag_sync)


def synchronization(data_source):

    _, osc_states, _, _ = load_sim_results(data_source)

    all_syncs, all_thetas = [], []
    for i in range(osc_states.shape[-1]):
        state = osc_states[:, :, i]
        r, theta = state_sync(state)
        all_syncs.append(r)
        all_thetas.append(theta)

    saveable = np.array([all_syncs, all_thetas])
    np.save(data_source.file_name('synchronizations', 'npy'), saveable, allow_pickle=False)
    return saveable


def source_synchronizations(data_source):
    data = source_data(data_source, 'synchronizations', synchronization)
    return data[0], data[1]


def plot_sync_time(data_source):

    syncs, thetas = source_synchronizations(data_source)
    time = load_sim_time(data_source)

    fig = plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(time, syncs)
    plt.xlabel('Time (s)')
    plt.ylabel('Synchronizations')
    plt.ylim([0, 1])

    plt.subplot(2, 1, 2)
    phases = np.abs(np.mod(thetas, 2 * np.pi) - np.pi)
    plt.plot(time, phases)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Phase')
    plt.ylim([-1.1*np.pi, 1.1*np.pi])
    plt.yticks([-np.pi, 0, np.pi], ['$-\pi$', '$0$', '$\pi$'])
    plt.tight_layout()

    plt.savefig(data_source.file_name('SynchronizationEvolution'))
    plt.close()



