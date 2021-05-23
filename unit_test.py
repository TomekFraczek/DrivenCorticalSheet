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
          osc.distance.flatten())

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
                                                                                                                     **kernel_params,
                                                                                                                     kn=kn)

    print(title)

def main():
    # distance_test(3,3)
    # wavelet_plot()
    # wavelet_test()
    # decouple_test()
    system(2,2)
    # gif_test()
    # normal_test()
    # move_dirs()
    # load_j()
    # index_ts()
    # plt_title()


if __name__ == '__main__':
    main()
    # build_ics(16,16)
    # spatial_kernel()
    # decouple()
