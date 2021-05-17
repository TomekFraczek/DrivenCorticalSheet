"""
boilerplate plot output directory handling and file name timestamp
plus common mpl.rcparams modifications for font and line size
"""
import os
import matplotlib as mpl
from datetime import datetime


class PlotSetup(object):
    def __init__(self, out_folder_name:str = 'plot_output'):
        self.title = out_folder_name
        self.directory = None
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S%f")
        self.file_path() # creates self.directory
        self.params()  # modify specific mpl.rcParams

    @staticmethod
    def clean(txt:str):
        ## TODO: fix w/ regex
        # print(txt)
        d = {"/":'-',
             "\\":'',
             '$':'',
             '[':'',
             ']':'',
             '(':'',
             ')':'',
             ',':''
             }

        """ '[()[\]{}] | [,\\$]'   ''   ''  """
        for (key, value) in d.items():
            txt = txt.replace(key,value)
        # cl = lambda t,d: t.replace(k,v) for (k,v) in d.items()
        # txt = cl(txt,d)
        return txt

    def plot_name(self,
                  txt:str=None,
                  extension:str = ''):
        p = ''
        if extension:
            p = '.'

        if not txt:
            txt = self.title

        ## TODO
        # txt = re.sub('[()[\]{}] | [\\$]','',txt)
        # txt = re.sub('/','-',txt)

        txt = self.clean(txt)
        file = ''.join((txt, '_', self.timestamp, p, f'{extension}'))

        return os.path.join(self.directory, file)

    def file_path(self):
        self.directory = os.path.join('plots', self.clean(self.title))
        os.makedirs(self.directory, exist_ok=True)

    def params(self):
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
