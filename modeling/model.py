import json

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from math import cos

from modeling.cortical_sheet import OscillatorArray
from modeling.wavelet import make_kernel
from modeling.interaction import Interaction

np.set_printoptions(precision=3, suppress=True)


class KuramotoSystem(object):
    def __init__(self, array_size, system_params, gain, initialize=True):

        print('Initializing model...')
        self.gain = gain

        self.interaction_params = system_params['interaction']
        self.interaction = Interaction(array_size, **self.interaction_params)

        self.kernel_params = system_params['kernel']
        self.wavelet_func = make_kernel('wavelet', **self.kernel_params)

        if initialize:  # Option to not initialize for later plotting purposes
            self.osc = OscillatorArray(array_size, system_params, gain)
            self.wavelet = self.wavelet_func(self.osc.distance)

        self.input_params = system_params['driver']
        self.external_input = self.input_params['use_driver']
        if self.external_input:
            self.n_inputs = array_size[0]
            self.input_weight = self.input_params['strength']
            self.input_freq = self.input_params['freq']
            self.input_effect = np.zeros((np.prod(array_size),))

        print('System ready!')

    def differential_equation(self, t: float, x: np.ndarray, ):
        """ of the form: wi - 'k/n * sum_all(x0:x_N)*fn_of_dist(xi - x_j) * sin(xj - xi))'
        """
        K = self.gain
        W = self.wavelet
        deltas = self.interaction.delta(x.ravel())

        G = self.interaction.gamma(deltas)

        N = np.prod(self.osc.ic.shape)

        dx = K/N*np.sum(W*G, axis=1).ravel() + self.osc.natural_frequency.ravel()

        if self.external_input:
            self.calc_input(t, x)
            dx += self.input_weight*self.input_effect

        print('t_step:', np.round(t, 4))

        return dx

    def solve(self,
              time_scale: tuple = (0, 10),
              ode_method: str = 'LSODA',  # 'Radau' works too, RK45 not so much
              continuous_fn=True,
              time_eval: np.ndarray = None,
              max_delta_t: float = 0.1,
              zero_ics: bool = False,
              ):
        """Solve ODE using methods, problem may be stiff so go with inaccurate to hit convergence
        """
        fn = self.differential_equation  # np.vectorize ?

        if not zero_ics:
            x0 = self.osc.ic.ravel()
        else:
            x0 = np.zeros(np.prod(self.osc.ic.shape))

        return solve_ivp(fn,
                         time_scale,
                         x0,
                         t_eval=time_eval,
                         max_step=max_delta_t,
                         method=ode_method,
                         dense_output=continuous_fn,
                         vectorized=False
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
    plt.pcolormesh(deltas.T, dists.T, interaction, cmap='coolwarm', shading='gouraud')
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
                "beta": 0,
                "r": 0
            },
            "kernel": {
                "s": 2,
                "width": 40
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
        }
    )
