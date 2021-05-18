"""
construct wavelet for distance decay spatial kernel
w = kernel(spatial_wavelet,x,*params.values(),True)
returns a normalized gaussian nth order derivative
"""
from plotting.plotformat import PlotSetup

import numpy as np
from matplotlib import pyplot as plt
from math import exp


def make_kernel(kernel_type, **params):
    if kernel_type == "constant":
        return constant
    elif kernel_type == "wavelet":
        def func(x):
            return wavelet(x, **params)
        return func


def constant(x):
    return np.ones(shape=x.shape)


def wavelet(x: np.ndarray, s: float, width: float) -> np.ndarray:
    """generalized kernel using gaussian paramaterization"""
    # Direct implemented
    x_p = np.pi * x / width
    k = exp(-0.5*s**2)
    wave = np.exp(-0.5*x_p**2) * (np.exp(1j*s*x_p) - k)
    norm = 1 - k

    return np.real(wave)/norm


def gauss_width(b: float = 0, c: float = 1, **kwargs):
    return np.linspace(b - 3.5 * c,  b + 3.5 * c, num=int(1e6))


def gaussian(
             x: np.ndarray,
             a: float = 1,
             b: float = 0,
             c: float = 1
             ) -> np.ndarray:
    """generalized gaussian function"""
    return a*np.exp(-(x-b)**2/2/c**2)


def plot_wavelet(X: np.ndarray,
                 plot_title:str = 'placeholder',
                 y_axis:str = 'y',
                 x_axis:str = 'x',
                 ):
    """plot the wave form for spatial kernel"""
    fmt = PlotSetup(plot_title)  # plotting format obj
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)

    ax.plot(X[...,0],X[...,1],'-b')

    plt.title(plot_title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(b=True, which='major', axis='both')
    plt.show()
    #fig.savefig(fmt.plot_name(plot_title,'png'))
    # plt.close('all')


def main():
    distance = 200
    resolution = 1000
    x = np.linspace(-distance, distance, resolution)
    s = 3
    wave = wavelet(x, s, distance)

    plot_wavelet(
        np.asarray([x, wave]).T,
        f'Morelt Wavelet with width={distance} and s={s}'
    )

if __name__ == '__main__':
    main()
