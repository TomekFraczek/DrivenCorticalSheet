from scipy.interpolate import (interp2d,
                               RectBivariateSpline,
                               SmoothBivariateSpline,
                               )
# from scipy.ndimage.filters import gaussian_filter
import numpy as np


def smooth_image(
                 z:np.ndarray,
                 x:np.ndarray,
                 y:np.ndarray,
                 scale:int = 2,
                 # sigma:np.ndarray = np.array([7,7])
                 )->np.ndarray:

    s = SmoothBivariateSpline(z,x,y)
    return s.__call__(x,y)
