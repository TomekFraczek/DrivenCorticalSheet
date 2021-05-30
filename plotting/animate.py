# -*- coding: utf-8 -*-
"""
make gif from selected folder
credit to https://github.com/dm20/gif-maker
"""
import imageio
import os
import re
import shutil
import numpy as np
from copy import deepcopy
from os.path import isfile, join
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from plotting.plotformat import PlotSetup
from progressbar import progressbar
from datetime import datetime

# gif_maker many thanks
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
# fp_in = "/path/to/image_*.png"
# fp_out = "/path/to/image.gif"
#
# # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
# img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
# img.save(fp=fp_out, format='GIF', append_images=imgs,
#          save_all=True, duration=200, loop=0)


class Animator(object):
    def __init__(self, model_config, dir_format: PlotSetup, fig_size=(10, 8)):
        self.dir_format = dir_format
        self.config = model_config

        self.temp = PlotSetup(base_folder=self.dir_format.directory, label='frames')
        self.fig_size = fig_size
        self.x_axis = 'Horizontal Node Location'
        self.y_axis = 'Vertical Node Location'
        self.title = self.make_title()
        self.resolution = 24

        self.shape = self.config["sqrt_nodes"], self.config["sqrt_nodes"]
        self.x = None
        self.y = None
        self.color_scale = None
        self.prep_space()

    def animate(self, oscillator_states, times, cleanup=True):
        """Create an animation of the evolution of oscillator phases over time"""
        self.plot_frames(oscillator_states, times)
        self.to_gif(True)  # sort by timestamp broke if True
        if cleanup:
            self.cleanup()

    def plot_frames(self, oscillator_states, times):
        """Plot the phases of all oscillators through time"""
        print('Preparing frames...')
        for k in progressbar(np.arange(oscillator_states.shape[2])):
            oscillators_now = oscillator_states[..., k]
            self.plot_frame(oscillators_now, times[k])

    def plot_frame(self, raw_phases, t):
        """Plot all the oscillator phases mapped onto abs((-pi, pi]) at a single point in time"""
        phases = np.abs(np.mod(raw_phases, 2 * np.pi) - np.pi).ravel()

        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        plt.tricontourf(
            self.x, self.y, phases,
            self.color_scale,
            cmap=plt.cm.nipy_spectral,
        )
        self.format_frame(ax)

        title = self.this_title(t)
        plt.title(title)
        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)

        plt.grid(b=None, which='major', axis='both')
        fig.savefig(self.temp.file_name(f'Phases_at_t={t:.2f}_{datetime.now().strftime("%y%m%d_%H%M%S%f")}', 'png'))
        plt.close('all')

    def to_gif(self, sort: bool = False, ext: str = 'png'):
        """Convert all the .png images in this dir into a gif, sorting by timestamp if requested"""
        print('Stitching into a gif...')
        file_list = [
            f for f in os.listdir(self.temp.directory)
            if isfile(join(self.temp.directory, f)) and f.endswith(ext)
        ]

        if sort:
            s = lambda x: re.split(r'_t=\d*\.*\d*_',str(x),1)   # t = 1.4_20200505... & 15_2020..
            index = np.array([s(file) for file in file_list], dtype=str)
            if len(index.shape) == 1:
                print('err 1D arry')
                return False

            files = np.array([index[..., -1], file_list], dtype=str).T
            files = files[files[..., 0].argsort()]
            file_list = list(files[:, 1])

        images = list(map(
            lambda f: imageio.imread(os.path.join(self.temp.directory, f)),
            file_list
        ))
        gif_name = self.dir_format.file_name('animation', 'gif')
        imageio.mimsave(gif_name, images, duration=self.config['frame_rate'])

    def cleanup(self):
        """Remove the contents of the temp directory (individual frames of the phase evolution gif)"""
        target_path = self.temp.directory

        for filename in os.listdir(target_path):
            file_path = os.path.join(target_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                return False

        try:
            os.rmdir(target_path)

        except OSError as err:
            print(err)
            return False
        return True

    def prep_space(self):
        # want to keep this dependence on init setup may also use z.shape[]
        x = np.linspace(0, self.shape[0], self.shape[1])
        y = np.linspace(0, self.shape[1], self.shape[0])

        x, y = np.meshgrid(x, y, sparse=False)
        self.x, self.y = x.ravel(), y.ravel()
        self.color_scale = np.linspace(0, np.pi, self.resolution, endpoint=True)

    def make_title(self):
        kn = self.config['gain_ratio']
        r = self.config['system']['interaction']['r']
        beta = self.config['system']['interaction']['beta']
        s = self.config['system']['kernel']['s']
        return f'R={r:.2f} $\\beta$={beta:.2f} K/N={kn:.2f} & s={s:.2f}'

    def this_title(self, t):
        title = deepcopy(self.title)
        if t > 10:
            title += f' at t = {t:.0f}'
        else:
            title += f' at t = {t:2.1f}'
        return title

    def format_frame(self, ax: plt.Axes):
        plt.gca().invert_yaxis()
        plt.grid(b=True, which='major', axis='both')

        plt.clim(self.color_scale[0], self.color_scale[-1])

        plt.colorbar(ticks=self.color_scale[::-1][::5], format='%1.2f')

        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=self.shape[0]/4))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=self.shape[1]/4))
        # plt.tight_layout()

# TODO enable zip img
    #     try:
    #         new_fldr = self.fmt.plot_name(str(targetpath.stem)).stem
    #         archive = self.plot_directory/new_fldr
    #         print('***target',targetpath.stem)
    #         print('***to archive',new_fldr)
    #         os.replace(targetpath,archive)
    #
    #         # self.zip(archive)
    #
    #     except:
    #         print(f'error timestamping image fldr, make sure to clean {targetpath.stem} up before running agin :)')
    #
    # def zip(self, dir:str):
    #     try:  # base_name, format[, root_dir[, base_dir
    #         shutil.make_archive(archive, 'zip', self.plot_directory)
    #         # os.rmdir(archive, *, dir_fd=None)
    #     except:
    #         print(f'error zipping images, make sure to clean {targetpath.stem} up before running agin :)')
