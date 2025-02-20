'''
Helper functions to make the grids used in HA models
'''

# load some packages
from toolkit.helpers import agg_dist
from scipy import sparse
import numpy as np

## --- Asset Grid ---
def asset_grid(N: int, min: float, max: float, p: float = 4) -> np.ndarray[float]:
    '''
    Creates an unevenly spaced asset grid with `N` points between `min`
    and `max`.
    '''
    # uneven spacing gets more accuract at poitns where theres more nonlinearity (near norrowing constraint)
    return min + np.linspace(0, (max - min)**(1 / p), N)**p


## --- Idiosyncratic Grid ---
def rouwenhorst(N: int, rho: float, sigma: float) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''
    Creates the Rouwenhorst matrix for an AR1 process with persistance rho,
    standard deviation sigma, and N gridpoints
    '''
    # config
    sigma_process = np.sqrt(sigma**2 / (1 - rho**2))  # standard deviation of whole ar1 process, not just movements
    pq = (1 + rho) / 2  # p = q
    psi = sigma_process * np.sqrt(N - 1)  # max/min gridpoint, finer grid that spreads out more as N increases
    grid = np.linspace(-psi, psi, N)  # to map to logs, the grid becomes exp(grid)

    # make matrix
    mat = np.array([
            [pq, 1-pq],
            [1-pq, pq],
        ])
    for _ in range(2, N):
        mat_l = mat
        mat = np.zeros((len(mat_l) + 1, len(mat_l) + 1))  # make matrix with 1 more dimension
        mat[:-1, :-1] += pq * mat_l  # top left
        mat[1:, :-1] += (1-pq) * mat_l  # bottom left
        mat[:-1, 1:] += (1-pq) * mat_l  # top right
        mat[1:, 1:] += pq * mat_l  # bottom right
        mat[:, 1:-1] /= 2  # sum to 1

    return grid, mat

def idiosyncratic_grid(N: int, rho: float, sigma: float) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    '''
    Creates the idiosyncratic grid using a logspaces Rouwenhorst process.
    Grid has `N` gridpoints, `rho` persistance, and `sigma` movement.

    Returns the grid, distribution, and transition matrix
    '''
    ## get the transition matrix
    grid, tran_mat = rouwenhorst(N, rho, sigma)

    ## solve for the distribution
    grid = np.exp(grid)
    dist = get_ss_tran_mat_dist(tran_mat)

    return grid, dist, tran_mat, agg_dist(dist, grid)


# --- Distribution ---
# get the steady state distubrution on the grid
def get_ss_tran_mat_dist(tran_mat):
    '''
    Gets the distribution for a transition matrix

    Uses sparse matrix algebra if its a sparse matrix 
    '''
    if sparse.issparse(tran_mat):  # use sparse algebra to get eigenvector
        vals, vecs = sparse.linalg.eigs(tran_mat, k=1)  # get dominant eigenvector
        i = 0
        dist = vecs.real
    else:  # use numpy
        vals, vecs = np.linalg.eig(tran_mat)
        i = np.abs(vals).argmax()  # get dominant eigenvalue
        dist = vecs.T[i].real
    assert np.isclose(vals[i], 1)  # markov matrix
    dist /= dist.sum()  # normalize

    return dist
