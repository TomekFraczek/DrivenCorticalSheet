import os
import imageio
import numpy as np
from skimage.transform import resize
from skimage.util import crop
from progressbar import progressbar
from plotting.animate import animate_one
from plotting.common import load_sim_results, source_data, load_sim_time, load_config, var_names, \
    get_point_id, get_rep_id
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


def get_animation_data(data_src):

    if not data_src.has_file('animation', 'gif'):
        animate_one(data_src)
    gif_reader = imageio.get_reader(data_src.file_name('animation', 'gif'))
    return gif_reader


def source_animation_data(data_src):

    config = load_config(data_src)
    shape = config['point_config']['x-var']['num'], config['point_config']['y-var']['num']
    out_data = np.zeros(shape).tolist()

    points = [f for f in data_src.sub_folders() if get_point_id(f)]
    for p in points:
        possible_reps = [f for f in p.sub_folders() if get_rep_id(f)]
        rep = np.random.choice(possible_reps)

        gif = get_animation_data(rep)
        index = np.unravel_index(int(get_point_id(p)), shape)
        out_data[index[0]][index[1]] = gif

    return out_data


def mega_gif(data_src):

    gif_readers = source_animation_data(data_src)

    sample = gif_readers[0][0]

    mega_writer = imageio.get_writer(data_src.file_name('mega_animation', 'gif'))
    for frame_i in progressbar(range(sample.get_length())):

        all_pts = []
        for reader_row in gif_readers:

            one_row = []
            for reader in reader_row:
                this_frame = reader.get_next_data()
                cropped = crop(this_frame, [(85, 75), (120, 240), (0, 0)])
                reduced = resize(cropped, (160, 160), preserve_range=True).astype(np.ubyte)
                one_row.append(reduced)

            stacked_row = np.hstack(one_row)
            all_pts.append(stacked_row)

        big_frame = np.vstack(all_pts)
        mega_writer.append_data(big_frame)


    for many in gif_readers:
        for r in many:
            r.close()
    mega_writer.close()



