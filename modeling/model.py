import json

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import colors
from math import cos

from modeling.cortical_sheet import OscillatorArray
from modeling.wavelet import make_kernel
from modeling.interaction import Interaction

np.set_printoptions(precision=3, suppress=True)


class KuramotoSystem(object):
    def __init__(self, array_size, system_params, gain, initialize=True, boundary=False, location=None):

        print('Initializing model...')
        self.gain = gain
        self.dims = array_size

        self.interaction_params = system_params['interaction']
        self.interaction = Interaction(array_size, **self.interaction_params)

        self.kernel_params = system_params['kernel']
        self.wavelet_func = make_kernel('wavelet', **self.kernel_params)

        if initialize:  # Option to not initialize for later plotting purposes
            self.osc = OscillatorArray(array_size, system_params, gain, boundary)
            self.wavelet = self.wavelet_func(self.osc.distance)

        self.input_params = system_params['driver']
        self.external_input = self.input_params['use_driver']
        if self.external_input:
            self.n_inputs = np.prod(self.dims)

            y_dist = (np.array(range(self.n_inputs)) + self.dims[1]) // (self.dims[1])

            self.input_weight = self.input_params['strength'] / y_dist ** 2
            self.input_freq = self.input_params['freq']
            self.input_effect = np.zeros((np.prod(array_size),))

        self.run_loc = location

        print('System ready!')

    def differential_equation(self, t: float, x: np.ndarray, inspect:bool=False):
        """ of the form: wi - 'k/n * sum_all(x0:x_N)*fn_of_dist(xi - x_j) * sin(xj - xi))'
        """
        K = self.gain
        W = self.wavelet # indx where < threshold
        deltas = self.interaction.delta(x.ravel())

        G = self.interaction.gamma(deltas)  # mask

        N = np.prod(self.osc.ic.shape)

        dx = K/N*np.sum(W*G, axis=1).ravel() + self.osc.natural_frequency.ravel()

        if self.external_input:
            self.calc_input(t, x)
            dx += self.input_effect

        # dx = np.mod(dx,2*np.pi)*np.sign(dx)

        # print('t_step:', np.round(t, 4))

        if not inspect:
            return dx

        else:
            print('\nics\n',self.osc.ic.ravel(),
                '\nx\n',x,
                '\ndistance\n', self.osc.distance,
                '\ndiff\n',deltas,
                '\nG\n',G,'\nW\n',W,
                '\nG*W\n',G*W,
                '\nsum(G*W)\n',np.sum(W*G,axis=1),
                '\nnatl frequency\n',self.osc.natural_frequency,
                '\ndx\n',dx,
                '\nmean+3sigma(dx)',
                np.round(np.mean(dx)+3*np.std(dx),3),
                '\nmean(dx)',np.round(np.mean(dx),3),
                '\nmean-3sigma(dx)',
                np.round(np.mean(dx)-3*np.std(dx),3),
                '\nstdev(dx)', np.round(np.std(dx),3),
                )
            # input('\n...')
            return dx

    def make_time_event(self, save_freq):
        output = self.run_loc.file_name('saved_state', 'npy')
        end_time = self.run_loc.file_name('save_time', 'npy')

        def time_event(t, y):
            dist = t - round(t, save_freq - 1)
            # print(f'  Nearest save is {round(dist, 8)}s away (t={round(t,8)})')
            if abs(dist) < 10**(-6):
                print(f'Saving state at t={t}')
                np.save(output, y, allow_pickle=False)
                np.save(end_time, t, allow_pickle=False)
            return dist

        return time_event

    def solve(self,
              time_scale: tuple = (0, 10),
              ode_method: str = 'LSODA',  # 'Radau' works too, RK45 not so much
              continuous_soln=True,
              time_eval: np.ndarray = None,
              max_delta_t: float = 0.1,
              min_delta_t: float = 0,
              zero_ics: bool = False,
              save_freq: int = 0
              ):
        """Solve ODE using methods, problem may be stiff so go with inaccurate to hit convergence
        """
        fn = self.differential_equation  # np.vectorize ?

        if not zero_ics:
            x0 = self.osc.ic.ravel()
        else:
            x0 = np.zeros(np.prod(self.osc.ic.shape))

        if save_freq and self.run_loc is not None:
            events = [self.make_time_event(save_freq)]
        else:
            events = None

        return solve_ivp(fn,
                         time_scale,
                         x0,
                         t_eval=time_eval,
                         max_step=max_delta_t,
                         min_step=min_delta_t,
                         method=ode_method,
                         dense_output=continuous_soln,
                         vectorized=False,
                         events=events
                         )

    def calc_input(self, t: float, all_phases: np.ndarray):
        osc_phases = all_phases[:self.n_inputs]
        input_phase = self.input_freq * t * 2*np.pi
        deltas = input_phase - osc_phases
        d_phase = self.input_weight * -1 * np.sin(deltas)
        self.input_effect[:self.n_inputs] = d_phase


def plot_existing_interaction(deltas, dists, system, out_fmt=None):
    diff_part = system.interaction.gamma(deltas)
    wave_part = system.wavelet_func(dists)

    interaction = np.zeros((len(diff_part), len(wave_part)))

    for i in range(len(diff_part)):
        for j in range(len(wave_part)):
            interaction[i, j] = (system.gain/len(dists)**2) * diff_part[i] * wave_part[j]

    deltas, dists = np.meshgrid(deltas, dists)

    plt.figure(figsize=(15, 12))
    divnorm = colors.TwoSlopeNorm(vmin=np.min(interaction), vmax=np.max(interaction), vcenter=0.0)
    plt.pcolormesh(deltas.T, dists.T, interaction, cmap='coolwarm', shading='gouraud', norm=divnorm)
    plt.colorbar()
    plt.title(f'Interaction term -- '
              f'  Gamma: {json.dumps(system.interaction_params)}'
              f'  Kernel: {json.dumps(system.kernel_params)} ')
    plt.xlabel('Phase Difference')
    plt.ylabel('Node Distance')

    if out_fmt is None:
        plt.show()
    else:
        plt.savefig(out_fmt.file_name('interaction', 'png'))


def plot_interaction(size, sys_params, gain_ratio=1, out_fmt=None):
    deltas = np.linspace(-np.pi, np.pi, 100)
    dists = np.linspace(-size/2, size/2, size+1)

    system = KuramotoSystem((size, size), sys_params, gain_ratio, initialize=False)
    plot_existing_interaction(deltas, dists, system, out_fmt)


if __name__ == "__main__":

    plot_interaction(
        100,
        {
            "interaction": {
                "beta": 0.6,
                "r": 0.25
            },
            "kernel": {
                "s": 2,
                "width": 8
            },
            "initial": {
                "type": "uniform",
                "low": 0,
                "high": 6.28318
            },
            "natural_freq": {
                "a": 1,
                "b": 0,
                "c": 0.4
            },
            "driver": {
                "use_driver": True,
                "strength": 2.0,
                "freq": 0.25
              }
        }
    )
