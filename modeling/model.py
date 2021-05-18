import numpy as np
from scipy.integrate import solve_ivp

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

        self.prev_t = 0

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

        if t - self.prev_t < 1e-6:
            print(f'Small timestep. dx stats are: mean {np.mean(dx)} stdev {np.std(dx)} min {np.min(dx)} max {np.max(dx)}')

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
