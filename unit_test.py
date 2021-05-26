"""
"""
import sys
from pathlib import Path
print(Path(__file__).resolve().parents[1])
sys.path.append(Path(__file__).resolve().parents[1])

import numpy as np
np.set_printoptions(precision=2, suppress=True)
from datetime import datetime as dt

#TODO refactor useful ideas within, for instance distance test needs more input params now

def distance_test(m:int = 128,
                  n:int = 128,
                  ):
    from modeling.cortical_sheet import OscillatorArray

    syst_params =  {
                  "initial": {
                    "type": "uniform",
                    "low": 0,
                    "high": 6.28318
                  },
                  "interaction" : {
                    "beta":0,
                    "r":0
                  },
                  "kernel" : {
                    "s":0,
                    "width":0
                  },
                  "normalize_kernel": False,
                  "natural_freq" : {
                    "a": 1,
                    "b":0,
                    "c":0.4
                  },
                  "driver": {
                    "use_driver": False,
                    "driver_weight":0
                  }
                }

    gain_ratio = 5

    domain = (-np.pi,np.pi)
    osc = OscillatorArray((m,n),syst_params,gain_ratio)

    print(dt.now(),#.strftime('%y%m%d_%H%M%S'),
          '\nics\n',
          osc.ic,
          '\n\ndistance shape\n',
          osc.distance.shape,
          '\n\ndistance vector\n',
          osc.distance)

    return osc.ic,osc.distance.flatten()

def wavelet_test():
    from modeling.wavelet import make_kernel

    _,y = distance_test(3,3)


    p = {
      "s":2,
      "width":2
    }


    s = make_kernel("wavelet",**p)


    w = s(y)
    print(dt.now(),'\nwavelet\n',w)



def decouple_test():
    from modeling.interaction import Interaction


    p = {'beta': 0.25, 'r':0.95}

    x,_ = distance_test(3,3)

    a = Interaction(x.shape,**p)

    y = a.delta(x.ravel())

    g = a.gamma(y)

    print(dt.now(),'\ngamma\n',g,
          '\n\nphase difference vector\n',
          y.flatten(),
          '\n\nmean difference vector\n',
          np.mean(y))

    return g.flatten()

def wavelet_plot():
    from modeling.wavelet import wavelet, plot_wavelet

    distance = 8
    resolution = 1000
    x = np.linspace(-distance, distance, resolution)
    s = 2.5
    wave = wavelet(x, s, distance)

    plot_wavelet(
        np.asarray([x, wave]).T,
        f'Morelt Wavelet with width={distance} and s={s}'
    )







def system(m,n):
    #initialize an osc array
    from modeling.cortical_sheet import OscillatorArray

    syst_params =  {
                  "initial": {
                    "type": "uniform",
                    "low": 0,
                    "high": 6.28318
                  },
                  "interaction" : {
                    "beta":0,
                    "r":0
                  },
                  "kernel" : {
                    "s":0,
                    "width":0
                  },
                  "normalize_kernel": False,
                  "natural_freq" : {
                    "a": 1,
                    "b":0,
                    "c":0.4
                  },
                  "driver": {
                    "use_driver": False,
                    "driver_weight":0
                  }
                }

    gain_ratio = 5

    osc = OscillatorArray((m,n),syst_params,gain_ratio)
    print(osc.ic)
    from modeling.wavelet import make_kernel

    # fixed time wavelet kernel

    p = {
      "s":2,
      "width":2
    }


    s = make_kernel("wavelet",**p)


    w = s(osc.distance)
    print(dt.now(),'\nwavelet\n',w)

    from modeling.interaction import Interaction


    q = {'beta': 0.25, 'r':0.95}


    a = Interaction(osc.ic.shape,**q)

    phase_difference = a.delta(osc.ic.ravel())
    g = a.gamma(phase_difference)

    print(dt.now(),
          '\nwavelet\n',
          w,'\n',type(w),
          '\n\nphase difference vector\n',
          g,'\n',
          type(g.flatten()),
          '\nwavelet*difference\n',
          w*g
          )
          # just going to renmae variable vals
    W = w

    deltas = phase_difference
    G = g


    N = np.prod(osc.ic.shape)
    K = gain_ratio*Ns

    print(K/N,'\n')

    print(W*G,np.sum(W*G,axis=1), np.sum(W*G,axis=1).shape,'\n')
    print(osc.natural_frequency.ravel(),'\n')
    dx = K/N*np.sum(W*G,axis=1) + osc.natural_frequency.ravel()
    print(dx, dx.shape,'\n')


def gif_test():
    from lib.animate import animate
    filepath = Path('/Users/Michael/Documents/GitHub/kuramoto-osc/Python/Oscillator Phase in 0_pi')
    vid = animate(filepath)
    vid.to_gif(filepath,0.75,True)



def move_dirs():
    from lib.plotformat import setup
    fmt = setup('test_dir',3)
    txt ='Oscillator Phase in pi'
    print(txt)
    print(fmt.plot_name(str(txt)))





def load_j():
    import json
    f = open('model_config.json')
    var = json.load(f)
    [print(var['test_set0'][k]) for k,v in var['test_set0'].items()]


def index_ts():
    zshape = (24,24,500)

    rng = np.random.default_rng()

    rnd_idx = rng.choice(np.arange(zshape[0]),
                         size=2,
                         replace=False,
                         )
    print(rnd_idx)
    idx = np.array(
    [[ 6,  1],
     [ 6, -1],
     [ 4,  1],
     [ 4, -1],
     [ 5,  1],
     [ 5, -1],
     [ 6 , 0],
     [ 4,  0]]
    )

    idl0 = np.where(idx[:,0]<=zshape[0])[0]
    idl1 = np.where(idx[:,1]<=zshape[1])[0]

    idz0 = np.where(idx[:,0]>=0)[0]
    idz1 = np.where(idx[:,1]>=0)[0]

    print(idl0,idl1,idz0,idz1)
    idu = np.intersect1d(idl0,idz0)
    idv = np.intersect1d(idl1,idz1)
    idw = np.intersect1d(idu,idv)

    print( idu, idv, idw, idx[idw,:])



"""
def plt_title():


    interaction_params:dict = {'beta': 0.75,'r': 0.25}
    kernel_params:dict = {'a': 10000/3*2,
                          'b': 0,
                          'c': 10, # breadth of wavelet
                          'order': 4}
    title=None
    domain = [0,np.pi]
    kn=11.1
    samples = 5
    if abs(domain[0]) % np.pi == 0 and not domain[0] == 0:
        ti = r'\pi'
        ti = '-'+ti
    else:
        ti = str(domain[0])

    if abs(domain[1]) % np.pi == 0 and not domain[1] == 0:
        tf = r'\pi'
    else:
        tf = str(domain[1])

    if not title:
        print(interaction_params,
                kernel_params,
                            )

        title = 'Timeseries for {s} Random Neighbors R={r:.2f} $\\beta$={beta:.2f} K/N={kn:.1f} & c={c:.0f})'.format(s=samples,
                                                                                                                     **interaction_params,

                                                                                                                     print(title)

"""




def ind_compare():
    x,y = np.meshgrid(np.arange(3),
                      np.arange(3),
                      sparse=False, indexing='ij')
    print('ij\nx:\n',x,'\n\ny:\n',y)

    x,y = np.meshgrid(np.arange(3),
                      np.arange(3),
                      sparse=False, indexing='xy')

    print('\nxy\nx:\n',x,'\n\ny:\n',y)


def torus(x, y, size_x, size_y):
    """
    https://stackoverflow.com/questions/62522809/\
    how-to-generate-a-numpy-manhattan-distance-array-with-torus-geometry-fast
    >>> f(x=2, y=3, size_x=8, size_y=8)
    array([[5, 4, 3, 2, 3, 4, 5, 6],
           [4, 3, 2, 1, 2, 3, 4, 5],
           [3, 2, 1, 0, 1, 2, 3, 4],
           [4, 3, 2, 1, 2, 3, 4, 5],
           [5, 4, 3, 2, 3, 4, 5, 6],
           [6, 5, 4, 3, 4, 5, 6, 7],
           [7, 6, 5, 4, 5, 6, 7, 8],
           [6, 5, 4, 3, 4, 5, 6, 7]])
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
    return np.add.outer(x_template, y_template)




def main():
    distance_test(3,3)
    # wavelet_plot()
    # wavelet_test()
    # decouple_test()
    # system(2,2)
    # gif_test()
    # normal_test()
    # move_dirs()
    # load_j()
    # index_ts()
    # plt_title()
    # print(torus(3,3,5,5))


if __name__ == '__main__':
    main()
    # build_ics(16,16)
    # spatial_kernel()
    # decouple()
