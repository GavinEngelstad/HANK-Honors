'''
Misc helper functions used in other classes
'''

# load some packages
import numpy as np
import numba as nb


def agg_dist(dist, grid):
    '''
    Takes a distirbution and values at each gridpoint of the distiribution
    and aggregates them 
    '''
    return (dist * grid).sum()


@nb.njit
def func_interp(x, xp, yp):
    '''
    Interpolate a function from xp to yp at points in x

    Essentially just passes everything to np.interp along the 0 axis
    '''
    N, imax = xp.shape  # dimensions
    y = np.empty((N, imax))  # end result into this array 
    for i in range(imax):
        y[:, i] = np.interp(x, xp[:, i], yp[:, i])

    return y


def add_or_insert(dict, key, value):
    '''
    Add value at key in dict. If key not in dict, inserts it in
    '''
    if key in dict:
        dict[key] = dict[key] + value  # use this instead of += to insert a new array, not modify existing one
    else:
        dict[key] = value
