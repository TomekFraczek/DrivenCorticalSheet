import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from modeling.cortical_sheet import OscillatorArray
from modeling.wavelet import wavelet, constant
from modeling.interaction import Interaction

np.set_printoptions(precision=3, suppress=True)


class KuramotoSystem(object):
    def __init__(self, array_size, system_params, gain,
                 external_input: bool = False, input_weight: float = 0, ):

        self.gain = gain
        self.kernel_params = system_params['kernel']
        self.interaction_params = system_params['interaction']

        self.osc = OscillatorArray(array_size, system_params, gain)

        self.wavelet = constant(self.osc.distance.ravel(), **self.kernel_params)

        self.interaction = Interaction(self.osc.ic.shape, **self.interaction_params)
        self.external_input = external_input
        self.input_weight = input_weight

    def differential_equation(self,
                              t:float,
                              x:np.ndarray,
                              ):
        """ of the form: xi - 'k/n * sum_all(x0:x_N)*fn_of_dist(xi - x_j) * sin(xj - xi))'
        """

        K = self.gain
        W = self.wavelet

        deltas = self.interaction.delta(
                    x.ravel()
                )
        G = (
            self.interaction.gamma(
                deltas
            )
        ).ravel()

        N = np.prod(self.osc.ic.shape)

        dx = K/N*np.sum(W*G) + self.osc.natural_frequency.ravel()

        if self.external_input:
            dx += self.input_weight*self.external_input_fn(t)

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

        """
        option to vectorize but need to change downstream, keep false
        """
        return solve_ivp(fn,
                         time_scale,
                         x0,
                         t_eval=time_eval,
                         max_step=max_delta_t,
                         method=ode_method,
                         dense_output=continuous_fn,
                         vectorized=False
                         )

    def external_input_fn(self, t:float):  # ,w:float):
        # cos(w*t)
        return 0


def plot_interaction():
    ratio = 0.5
    size = 90
    s = 2
    width = 40
    r = 0,
    beta = 0
    deltas = np.linspace(-np.pi, np.pi, 100)
    dists = np.linspace(-size/2, size/2, size+1)

    interact = Interaction((size, size), r=r, beta=beta)
    diff_part = interact.gamma(deltas)

    wave_part = wavelet(dists, s=s, width=width)

    interaction = np.zeros((len(diff_part), len(wave_part)))

    for i in range(len(diff_part)):
        for j in range(len(wave_part)):
            interaction[i, j] = ratio * diff_part[i] * wave_part[j]

    deltas, dists = np.meshgrid(deltas, dists)

    plt.figure()
    plt.pcolormesh(deltas.T, dists.T, interaction, cmap='coolwarm', shading='gouraud')
    plt.colorbar()
    plt.title(f'Interaction term: [r={r}, $\\beta$={beta}], [s={s}, width={width}] ')
    plt.xlabel('Phase Difference')
    plt.ylabel('Node Distance')

    # Tomek: apparently I don't understand how tricontourf works...
    # deltas = deltas.ravel()
    # dists = dists.ravel()
    # interacts = interaction.ravel()
    # plt.figure()
    # plt.tricontourf(
    #     dists, deltas,  interacts,
    #     cmap=plt.cm.get_cmap('coolwarm'),
    # )
    # plt.colorbar()
    # plt.xlabel('Phase Difference')
    # plt.ylabel('Node Distance')

    plt.show()


if __name__ == "__main__":
    plot_interaction()
