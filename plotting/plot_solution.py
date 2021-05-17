"""takes select soscillatorArray: "osc" instance from model.py and constructs plot
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from plotting.plotformat import PlotSetup

import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.interpolate import RectBivariateSpline


def save_data(data: np.ndarray, file_name: str = 'model_data'):
    print(file_name)
    fmt = PlotSetup(file_name)
    np.save(fmt.plot_name(file_name, 'npy'), data)


def plot_output(model, osc,
                data: np.ndarray,
                time: np.ndarray,
                samples: int = 4,
                seconds: int = 4,
                scale: bool = False,
                file_name: str = 'model_data'):

    # print(data.shape)
    for k in np.arange(data.shape[2]):
        data_here = data[...,k]
        model.plot_contour(osc, data[...,k], time[k],)

        if samples and np.round(100*time[k])%100==0 and not time[k]==time[-1]:
            print(np.round(time[k]))
            idx=np.where(time>=time[k])[0]  # larger set of two
            idy=np.where(time<time[k]+seconds)[0]
            idz = idx[np.in1d(idx, idy)]  # intersection of sets

            model.plot_timeseries(osc,
                                  data[...,idz],
                                  time[idz],
                                  samples,
                                  seconds)

################################################################################


def plot_timeseries(osc,
                    z:np.ndarray,
                    t:np.ndarray,
                    samples:int=3,
                    seconds:int = 2,
                    title:str = None,
                    y_axis:str = '$\cos(\omega*t)$',
                    x_axis:str = 'time, s',
                   ):
    """plot the solution timeseries for a random cluster of neighbor nodes
    """
    if not title:
        title = f'Solution Timeseries for {samples} Random Neighbors'

        if t[0]:
            if t[0]>10:
                title+=f' at t = {t[0]:.0f} to {t[-1]:.0f}'
            else:
                title+=f' at t = {t[0]:2.1f} to {t[-1]:2.1f}'

    fmt = PlotSetup(title)  # plotting format osc
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    rng = np.random.default_rng()

    rnd_node = rng.choice(np.arange(z.shape[0]),
                          size=2,
                          replace=False,
                          )

    # TODO generalize this to larger square m*n
    neighbors = np.array([[1,1,-1,-1,0,0,1,-1],
                          [1,-1,1,-1,1,-1,0,0]]).T

    idx = np.broadcast_to(rnd_node,neighbors.shape) + neighbors

    ##validate in range, since these are 2d but coupled pairs and where returns 1d just use unique
    # idlimit = np.where(idx<=z.shape[0:2])[0]
    # idzero = np.where(idx>=0)[0]

    # indx within limit
    idlimit0 = np.where(idx[:,0]<z.shape[0])[0]
    idlimit1 = np.where(idx[:,1]<z.shape[1])[0]
    # indx >0, actually if ~-1, -2 that is permissable but it won't be as local
    idzero0 = np.where(idx[:,0]>=0)[0]
    idzero1 = np.where(idx[:,1]>=0)[0]
    # down select x's, y's indiv
    idu = np.intersect1d(idlimit0,idzero0)
    idv = np.intersect1d(idlimit1,idzero1)
    # intersection of permissable sets
    idw = np.intersect1d(idu,idv)

    rnd_near = rng.choice(idx[idw,:],
                         size=samples,
                         replace=False,
                         )
    # rnd_near = np.squeeze(rnd_near)
    ax.plot(t,np.cos(z[rnd_node[0],rnd_node[1],:]),
            '-k',label=f'oscillator node ({rnd_node[0]},{rnd_node[1]})')



    colors = {0:'--r',
              1:'--b',
              2:'--g',
              3:'--c',
              4:'--m',
              5:'--y',
              }
    for k in np.arange(rnd_near.shape[0]):
        ax.plot(t,np.cos(z[rnd_near[k,0],rnd_near[k,1],:]),
                colors[k%len(colors.values())],
                label=f'node ({rnd_near[k,0]},{rnd_near[k,1]})')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(b=True, which='major', axis='both')
    ax.legend(loc=0)

    # plt.show()
    fig.savefig(fmt.plot_name(title,'png'))
    plt.close('all')


################################################################################

def plot_contour(osc,
                 z: np.ndarray,
                 t: float = None,
                 title: str = None,
                 scale: int = None):
    """
    """
    # want to keep this dependence on init setup may also use z.shape[]
    x = np.linspace(0, osc.ic.shape[0], osc.ic.shape[1])
    y = np.linspace(0, osc.ic.shape[1], osc.ic.shape[0])

    if scale:
        s = RectBivariateSpline(x, y, z,
                                kx=osc.ic.shape[0],
                                ky=osc.ic.shape[1])

        x = np.linspace(0, osc.ic.shape[0], osc.ic.shape[1]*scale)
        y = np.linspace(0, osc.ic.shape[1], osc.ic.shape[0]*scale)
        z = s(x, y)

    X, Y = np.meshgrid(x, y, sparse=False)

    #### this z%pi is to show phase locking and not precisely the phase state in time
    #### see figures in paper for demo
    #### may do %2pi but would look more chaotic

    phases = np.abs(np.mod(z, 2*np.pi) - np.pi)
    phase_array = np.asarray([X.ravel(), Y.ravel(), phases.ravel()]).T
    # print(phase_array.shape)

    if not title:
        kn = np.round(osc.gain / np.prod(osc.ic.shape))
        r = osc.interaction_params['r']
        beta = osc.interaction_params['beta']
        s = osc.kernel_params['s']
        title = f'R={r:.2f} $\\beta$={beta:.2f} K/N={kn:.0f} & s={s:.0f}'

        if t or not (t == None):
            if t>10:
                title+=f' at t = {t:.0f}'
            else:
                title+=f' at t = {t:2.1f}'

    plot_phase(osc,
               phase_array,
               title,
               'Vertical Node Location',
               'Horizontal Node Location'
               )



def plot_phase(osc,
               X,
               plot_title:str = None,
               y_axis:str = 'y',
               x_axis:str = 'x',
               resolution:int = 24
               ):
    """
    """
    folder = re.sub(r'(( at t = \d*\.\d*)|( at t = \d+))', '', plot_title)

    # print('\n***plt dir befor setup',folder)

    fmt = PlotSetup(folder)
    osc.plot_directory = fmt.directory


    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    colorscale = np.linspace(0, np.pi, resolution, endpoint=True)
    plt.tricontourf(X[..., 0], X[..., 1], X[..., 2],
                    colorscale,
                    cmap=plt.cm.nipy_spectral,
                    )

    plt.gca().invert_yaxis()
    plt.grid(b=True, which='major', axis='both')

    plt.clim(colorscale[0],colorscale[-1])


    # tick_marks = np.append(colorscale[::-1][::5],0).sort()

    plt.colorbar(ticks=colorscale[::-1][::5],format='%1.2f')

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=osc.ic.shape[0]/4))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=osc.ic.shape[1]/4))


    plt.title(plot_title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.grid(b=None, which='major', axis='both')
    # plt.show()
    fig.savefig(fmt.plot_name(plot_title,'png'))
    plt.close('all')

################################################################################
