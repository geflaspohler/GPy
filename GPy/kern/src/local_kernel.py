import GPy
import numpy as np
import math
import copy
from .stationary import Stationary
from .psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from ...core import Param
from paramz.transformations import Logexp
from .grid_kerns import GridRBF

class Local(Stationary):
    """
    Local kernel 
    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name = 'local', useGPU = _support_GPU, inv_l = False):
        super(Local, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU = useGPU)

    def K_of_r(self, r):
        for i in xrange(r.shape[0]):
            for j in xrange(r.shape[1]):
                element = r[i, j]
                if element < 2.0 * math.pi:
                    r[i, j] = ((2.0 * math.pi - element) * (1 + math.cos(element) * 0.5) + 1.5 * math.sin(element)) / (3.0 * math.pi)
                else:
                    r[i, j] = 0
        return r
            


