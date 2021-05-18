"""
boilerplate plot output directory handling and file name timestamp
plus common mpl.rcparams modifications for font and line size
"""
import os
import matplotlib as mpl
from slugify import slugify
from datetime import datetime


class PlotSetup(object):
    def __init__(self, base_folder='plots', label='', readonly=False):
        self.base = base_folder
        self.label = label
        self.directory = None
        self.timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
        self.readonly = readonly

        self.make_file_path()  # creates self.directory
        self.set_mpl_params()  # modify specific mpl.rcParams

        if readonly:
            print(f'Ready to LOAD from: {self.directory}')
        else:
            print(f'Ready to SAVE OUT to: {self.directory}')

    @staticmethod
    def clean(text: str):
        return text     # we trust our users right? lol   slugify(text)

    def file_name(self, title: str, extension: str = ''):

        p = '.' if extension else ''

        txt = self.clean(title)
        file = ''.join((txt, p, f'{extension}'))

        return os.path.join(self.directory, file)

    def make_file_path(self):
        """Safely build a directory including the timestamp only if necessary"""
        self.directory = os.path.join(self.base, self.label)
        try:
            os.makedirs(self.directory, exist_ok=self.readonly)
        except FileExistsError:
            self.label = f'{self.label}_{self.timestamp}'
            self.make_file_path()

    @staticmethod
    def set_mpl_params():
        """matplotlib parameters plot formatting"""
        mpl.rcParams['axes.labelsize'] = 21
        mpl.rcParams['axes.titlesize'] = 17
        mpl.rcParams['xtick.labelsize'] = 22
        mpl.rcParams['ytick.labelsize'] = 22
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0
        mpl.rcParams['lines.linewidth'] = 2.8
        mpl.rcParams['lines.markersize'] = 18
        mpl.rcParams['lines.markeredgewidth'] = 3
        mpl.rcParams['legend.framealpha'] = 0.93
        mpl.rcParams['legend.fontsize'] = 20

    # ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g $\pi$'))
    # ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1,5,8]))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([1,5,8]))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
