import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from plotting.plotformat import PlotSetup

import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)


def plot_output(data: np.ndarray,
                time: np.ndarray,
                samples: int = 4,
                seconds: int = 4,
                scale: bool = False,
                file_name: str = 'model_data'):

    for k in np.arange(data.shape[2]):

        if samples and np.round(100*time[k])%100==0 and not time[k]==time[-1]:
            print(np.round(time[k]))
            idx=np.where(time>=time[k])[0]  # larger set of two
            idy=np.where(time<time[k]+seconds)[0]
            idz = idx[np.in1d(idx, idy)]  # intersection of sets

            plot_timeseries(data[..., idz], time[idz], samples, seconds)

################################################################################


def plot_timeseries(z: np.ndarray,
                    t:  np.ndarray,
                    samples: int = 3,
                    seconds: int = 2,
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
