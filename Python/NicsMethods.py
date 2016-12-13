# Functions designed to make life easier

import numpy as np
import matplotlib.pyplot as plt


def imagesc(data):
    x = np.linspace(-100, -10, 10)
    y = np.array([-8, -3.0])

    data = np.random.randn(y.size,x.size)

    plt.imshow(data, aspect='auto', interpolation='none',
               extent=extents(x) + extents(y), origin='lower')

    # plt.savefig('py.png')



def extents(f):
    # Used in imagesc
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


