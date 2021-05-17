import sys
from scipy.integrate import solve_ivp
from datetime import datetime as dt

import numpy as np
np.set_printoptions(precision=3, suppress=True)


from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1])
if __name__ == '__main__' and __package__ is None:
    __package__ = 'kurosc'

from cortical_sheet import OscillatorArray
from wavelet import wavelet, constant
from interaction import Interaction
from lib.plot_solution import (plot_contour,
                               plot_timeseries)


class KuramotoSystem(object):
    def __init__(self, array_size, system_params, gain,
                 external_input: bool = False, input_weight: float = 0, ):

        self.gain = gain
        self.kernel_params = system_params['kernel']
        self.interaction_params = system_params['interaction']

        self.osc = OscillatorArray(array_size, system_params, gain)

        self.wavelet = constant(self.osc.distance.ravel(), **self.kernel_params)

        self.interaction = Interaction(self.osc.ic.shape, **self.interaction_params)
        self.plot_contour = plot_contour
        self.plot_timeseries = plot_timeseries
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


###############################################################################
## unit tests may need update below but not called into model
###############################################################################
def test_case():
    #initialize an osc array
    dimension = (2,2)
    domain = (0,np.pi)
    osc = OscillatorArray(dimension, domain)

    # fixed time wavelet kernel
    kernel_params = {'a': 10000/3*2,
                     'b': 0,
                     'c': 10,
                     'order': 4,
                     }
    interaction_params = ({'beta': 0, 'r':0},
                          {'beta': 0.25, 'r':0.95})

    w = wavelet(osc.distance.ravel(), **kernel_params)

    # test case using initial conditions
    a = Interaction(osc.ic.shape, **interaction_params[0])
    phase_difference = a.delta(osc.ic)
    g = a.gamma(phase_difference)

    print(dt.now(),
          '\nwavelet\n',
          w,'\n',type(w),
          '\n\nphase difference vector\n',
          g.ravel(),'\n',
          type(g.ravel()),
          '\nwavelet*difference\n',
          (w*g.ravel()).shape
          )




def run():
    nodes = 128
    time =  10
    kernel_params = {'a': 10000/3*2,
                     'b': 0,
                     'c': 1,
                     'order': 4,
                     }
    interaction_params = ({'beta': 0, 'r':0},
                          {'beta': 0.25, 'r':0.95}
                          )

    kuramoto = KuramotoSystem((nodes,nodes),
                                kernel_params,
                                interaction_params[0]
                                )
    solution = kuramoto.solve((0,time))
    """
    python -c "import numpy as np;
    a=np.array([[[2,3],[1,4]],[[0,1],[1,0]],[[6,7],[4,5]]]);
    b = a.flatten(); print(b);
    print(a,a.shape,'\n\n',b.reshape(3,2,2))"
    """
    osc_state = solution.y.reshape((solution.t.shape[0],nodes,nodes))%np.pi
    print(solution.y.shape, solution.t.shape)
    # print(osc_state[0])
    kuramoto.plot_solution(osc_state[-1],solution.t[-1])

if __name__ == '__main__':
    # test_case()
    run()


    """
    system of m*n indep variables subject to constraints:
    x - xij for all other ij to feed into sin (phase diff)
    w(m,n) for all other ij distances calculate once
    w(m,n) is distance dependent scalar but may be calculated more than once
    idea to evolve w(m,n) kernel function with time or provide feedback just for fun
    as if derivative power (n) or breadth of wave (a,c) or strength is modulated with system state
    """
