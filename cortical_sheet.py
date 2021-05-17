"""
construct 2d array of pase state
distance array

"""
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1])
if __name__ == '__main__' and __package__ is None:
    __package__ = 'kurosc'

import numpy as np
from lib.plot_solution import plot_phase
from wavelet import gaussian, gauss_width


class OscillatorArray(object):
    def __init__(self,  dimension: tuple, system_params, gain):
        self.init_params = system_params['initial']
        self.freq_params = system_params['natural_freq']

        self.ic = np.linspace(0, 2 * np.pi, dimension[0] * dimension[1]).reshape(dimension)  # self.prep_initial_conditions(*dimension)
        # Useful debug initital conditions that neatly increase over the whole space
        # np.linspace(0, 2 * np.pi, dimension[0] * dimension[1]).reshape(dimension)

        self.natural_frequency = self.prep_natural_frequency()
        self.distance = self.prep_distance()

        self.plot_phase = plot_phase
        self.plot_directory = None  # initialized in a plot module i think

        # for labeling plots:
        self.interaction_params = system_params['interaction']
        self.kernel_params = system_params['kernel']

        self.gain = gain

    def prep_initial_conditions(self, m: int = 16, n: int = 16) -> np.ndarray:
        """Randomly draw the initial phase for each oscillator in the array"""
        params = self.init_params
        rng = np.random.default_rng()
        if params['type'] == 'gaussian':
            x = gauss_width(*params)
            prob = gaussian(x, *params.values())
            prob = prob/np.sum(prob)  # pdf for weights
            phase = rng.choice(x, size=m*n, p=prob, replace=False)\
                .reshape(m, n)
        elif params['type'] == 'uniform':
            phase = rng.uniform(size=m*n, low=params['low'], high=params['high'])\
                .reshape(m, n)
        else:
            raise KeyError('Invalid initial conditions!')

        print(f'\nintial contitions ({params["type"]}) in phase space:',
              np.round(np.mean(phase), 3),
              '\nstdev:',
              np.round(np.std(phase), 3)
              )

        return phase

    def prep_natural_frequency(self) -> np.ndarray:
        """rtrn x vals for normal weighted abt 0hz
            #  distinct vals for replace = false
        """

        params = self.freq_params

        x = gauss_width(**params)
        weights = gaussian(x, **params)
        prob = weights/np.sum(weights)  # pdf for weights

        rng = np.random.default_rng()
        frequency = rng.choice(x, size=np.prod(self.ic.shape), p=prob, replace=True)

        print('\nmean natural frequency in hz:',
              np.round(np.mean(frequency),3),
              '\nstdev:',
              np.round(np.std(frequency),3),
              '\nconverted to phase angle on output'
              )
        return frequency*np.pi*2

    def prep_distance(self, t: str = 'float') -> np.ndarray:

        """construct m*n*(m*n) array of euclidian distance as integer or float
           this could be optimized but is only called once as opposed to eth phase difference calc
        """
        d = np.zeros([self.ic.shape[0]*self.ic.shape[1],
                      self.ic.shape[1]*self.ic.shape[0]])

        u,v = np.meshgrid(np.arange(self.ic.shape[0]),
                          np.arange(self.ic.shape[1]),
                          sparse=False, indexing='xy')
        u = u.ravel()
        v = v.ravel()
        z = np.array([u,v])

        for (k,x) in enumerate(z):
            d[k,:] = np.array(np.sqrt((u - x[0])**2 + (v - x[1])**2),dtype=t)
        return d


def main():
    """this demos a random contour plot"""
    cortical_array = OscillatorArray((64, 64), (-np.pi, np.pi), 1)
    x = np.linspace(0, cortical_array.ic.shape[0], cortical_array.ic.shape[1])
    y = np.linspace(0, cortical_array.ic.shape[1], cortical_array.ic.shape[0])
    x, y = np.meshgrid(x, y)

    phase_array = np.asarray([x.ravel(),
                              y.ravel(),
                              cortical_array.ic.ravel()]
                              ).T

    cortical_array.plot_phase(phase_array,
                             'Oscillator Phase $\in$ [-$\pi$,$\pi$)',
                             'Location y',
                             'Location x'
                             )

if __name__ == '__main__':
    main()
