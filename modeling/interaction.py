"""
second order neural interrupting interactions
interaction is class, gamma is function, delta function

if need to run in here pull in plotting folder to dir
"""
from plotting.plotformat import PlotSetup

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# As a reference: Eqn 13 + Fig 4


class Interaction(object):
    def __init__(self, dim, **params):
        self.dimension = dim
        self.beta = params['beta']
        self.r = params['r']

    def delta(self, phase_array) -> np.ndarray:
        """Phase difference of element from global array"""

        d = np.zeros([self.dimension[0]*self.dimension[1],
                      self.dimension[1]*self.dimension[0]])

        # TODO validate this index assignment with ravel()
        for (k, p) in enumerate(phase_array):
            d[k, :] = np.array((phase_array - p), dtype=float)

        # The above should be logically equivalent to the below
        # k=0
        # for j in np.arange(phase_array.shape[1]):
        #     for i in np.arange(phase_array.shape[0]):
        #         # print(i*j,j,i)
        #         d[k,...] = (phase_array - phase_array[i,j]) % np.pi
        #         k+=1

        return d

    def gamma(self, x: np.ndarray) -> np.ndarray:
        sign = -1 # if self.r else 1  #togles between eqn 13 & 11 w/ presence of 2nd order term
        return sign*np.sin(x+self.beta) + self.r*np.sin(2*x)

    def plot_phase(self,
                   X: np.ndarray,
                   plot_title: str = 'placeholder',
                   y_axis: str = 'y',
                   x_axis: str = 'x',
                   ):
        fmt = PlotSetup(plot_title)  # plotting format obj
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        ax.plot(X[..., 0]/np.pi, X[..., 1],'-b')
        ax.plot(np.asarray([X[0, 0], X[-1, 0]])/np.pi, [0, 0], '-k', linewidth=1)

        plt.title(plot_title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g $\pi$'))
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1))
        plt.grid(b=True, which='major', axis='both')
        plt.show()
        # fig.savefig(fmt.plot_name(plot_title,'png'))
        # plt.close('all')


def main():
    """plot needs plotting in this wd
    """

    params = ({'beta': 0.0, 'r': 0.0},
              {'beta': 0.25, 'r': 0.95},
              {'beta': 0.25, 'r': 0.8},
              {'beta': 0.25, 'r': 0.7},
              )
    x = np.linspace(-np.pi, np.pi, 1000)

    for p in params:
        a = Interaction((1, 1),  **p)
        g = a.gamma(x)
        a.plot_phase(np.asarray([x,g]).T,
                    'R = {0}'.format(p['r']),
                    '',
                    # r'$\frac{d\theta}{dt}$',
                    r'$\phi$')


if __name__ == '__main__':
    main()
