"""
construct 2d array of phase state
distance array

"""

import numpy as np
from modeling.wavelet import gaussian, gauss_width

class OscillatorArray(object):
    def __init__(self,  dimension: tuple, system_params, gain, boundary=False):
        print(f'Initializing {dimension} oscillator array...')
        self.init_params = system_params['initial']
        self.freq_params = system_params['natural_freq']
        self.boundary = boundary
        self.gain = gain

        self.ic = self.prep_initial_conditions(*dimension)
        # Useful debug initital conditions that neatly increase over the whole space
        #     np.linspace(0, 2 * np.pi, dimension[0] * dimension[1]).reshape(dimension)

        self.natural_frequency = self.prep_natural_frequency()
        self.distance = self.prep_distance()


    def prep_initial_conditions(self, m: int = 16, n: int = 16) -> np.ndarray:
        """Randomly draw the initial phase for each oscillator in the array"""
        params = self.init_params
        rng = np.random.default_rng()
        if params['type'] == 'gaussian':
            # params[abc]
            x = gauss_width(**params)
            prob = gaussian(x, **params)
            prob = prob/np.sum(prob)  # pdf for weights
            phase = rng.choice(x, size=m*n, p=prob, replace=False)\
                .reshape(m, n)

        elif params['type'] == 'uniform':
            # params['low,high']
            phase = rng.uniform(size=m*n, low=params['low'], high=params['high'])\
                .reshape(m, n)
        else:
            raise KeyError('Invalid initial conditions!')

        print(f'    initial conditions ({params["type"]}) in phase space:\n '
              f'        mean: {np.round(np.mean(phase), 3)}\n'
              f'        st dev: {np.round(np.std(phase), 3)}')

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

        print(f'    natural frequency stats in hz:\n'
              f'        mean: {np.round(np.mean(frequency),3)}\n'
              f'        st dev: {np.round(np.std(frequency),3)}\n'
              f'    converted to phase angle on output')
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
        z = np.array([u,v]).T

        for (k,x) in enumerate(z):
            if not self.boundary:
                d[k,:] = np.array(np.sqrt((u - x[0])**2 + (v - x[1])**2),dtype=t)

            else:
                d[k,:] = self.torus(x[0],x[1],
                                    self.ic.shape[0],
                                    self.ic.shape[1]
                                    ).ravel()

        return d




    def torus(self,x, y, size_x, size_y):
        """
        https://stackoverflow.com/questions/62522809/\
        how-to-generate-a-numpy-manhattan-distance-array-with-torus-geometry-fast

        >>> f(x=1, y=1, size_x=3, size_y=3)
        array([[2, 1, 2],
               [1, 0, 1],
               [2, 1, 2]])
        """
        a, b = divmod(size_x, 2)
        x_template = np.r_[:a+b, a:0:-1] # [0 1 2 1] for size_x == 4 and [0 1 2 2 1] for size_x == 5
        x_template = np.roll(x_template, x) # for x == 2, size_x == 8: [2 1 0 1 2 3 4 3]
        a, b = divmod(size_y, 2)
        y_template = np.r_[:a+b, a:0:-1]
        y_template = np.roll(y_template, y)

        d = np.sqrt(np.add.outer(x_template**2, y_template**2))
        return d
