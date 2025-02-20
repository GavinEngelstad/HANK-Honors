'''
Handle irfs and simulations for models using the G matrix.
'''

## load some packages
from toolkit.helpers import add_or_insert
import numpy as np


## --- Policy Functions ---
# this could be njited, but testing shows that 2-5x slower
def var_policy_irf(dx, dX):
    '''
    Takes an irf for an aggregate X (dX) and a jacobian for a policy rule x
    with respect to X, returns the irf for x.
    '''
    ## intialize
    dxdZ = np.empty_like(dx)
    T, = dX.shape

    ## equation 88
    for t in range(T):
        dxdZ[t] = (dx[:T - t].T * dX[t:]).sum(axis=-1).T

    return dxdZ

def policy_irf(Xdx, G, Z_irf, shock):
    '''
    Get the aggregate polciy irf
    '''
    # setup
    irf = np.zeros_like(next(iter(Xdx.values())))

    # loop over variables
    for X, dx in Xdx.items():
        if X == shock:  # direct effects
            irf += var_policy_irf(dx, Z_irf)
        if X in G:  # indirect effects
            irf += var_policy_irf(dx, G[X] @ Z_irf)

    return irf

def decompose_policy_irf(Xdx, G, exog_irf, shock):
    '''
    Decompose a policy funcction impulse response
    '''
    # setup
    decomposition = {}

    # loop over possible effects
    for X, dx in Xdx.items():
        if X == shock:  # direct effects
            add_or_insert(decomposition, X, var_policy_irf(dx, exog_irf))
        if X in G:  # indirect effects
            add_or_insert(decomposition, X, var_policy_irf(dx, G[X] @ exog_irf))
    
    return decomposition


## --- Shocks ---
def ar_shock_irf(rho, T, sigma=1):
    '''
    Returns the T-long irf for a shock to an AR(1) process with persistance
    rho
    '''
    return sigma * rho**np.arange(T)


## --- General IRFs ---
# these are slightly faster than the rifs function, but are order determinate
def single_shock_irfs(G, rho, T, sigma=1, dxs=None, shock=None):
    '''
    Returns the irfs for endogenous variable given a single AR1 shock response
    matrix G

    If dx is defined, also gets the policy function irfs (takes much longer)
    '''
    # get the shock irf
    Z_irf = ar_shock_irf(rho, T, sigma)

    # get the irfs for each aggregate
    endog_irfs = {X: GX @ Z_irf for X, GX in G.items()}

    ## policy function irfs
    if dxs is not None:
        for x, Xdx in dxs.items():
            endog_irfs[x] = policy_irf(Xdx, G, Z_irf, shock)

    return endog_irfs, Z_irf

def all_shock_irfs(G, rho, T, sigma=1, dxs=None):
    '''
    Returns the irfs for all engoenous varibales given all possible AR1 shocks
    '''
    # if rho is a number, turn it into a dict
    if not isinstance(rho, dict):
        rho = {Z: rho for Z in G.keys()}
    if not isinstance(sigma,dict):
        sigma = {Z: sigma for Z in G.keys()}
    
    # calcualte irfs
    all_irfs = {Z: single_shock_irfs(G[Z], rho[Z], T, sigma=sigma[Z], dxs=dxs, shock=Z)[0] for Z in G.keys()}

    return all_irfs


## --- Decompositions ---
def decompose_irf(G, int_block_jacs, exog_irf, shock):
    '''
    Decompose an irf into the different direct effects into the intermediate step

    Ex: Krusell Smith C is decomposed into R and W effects
    '''
    ## setup
    decomposition = {}

    ## loop over possible effects
    for X in int_block_jacs.keys():
        if X == shock:
            decomposition[X] = int_block_jacs[X] @ exog_irf  # direct shock effects
        elif X in G.keys():
            decomposition[X] = int_block_jacs[X] @ (G[X] @ exog_irf)  # indirect other effects
    
    return decomposition

def decompose_single_shock_irfs(G, int_block_jacs, exog_irf, shock, dxs=None):
    '''
    Decompose all the irfs for a single shock into the direct effects into the intermediate
    step
    '''
    # setup
    decomposition = {}

    # loop over affectsed variables
    for X in G.keys():
        if X not in int_block_jacs:
            continue
        decomposition[X] = decompose_irf(G, int_block_jacs[X], exog_irf, shock)
    
    # dxs decomposition
    if dxs is not None:
        for x, Xdx in dxs.items():
            decomposition[x] = decompose_policy_irf(Xdx, G, exog_irf, shock)
        
    return decomposition

def decompose_all_irfs(G, int_block_jacs, exog_irf, shocks, dxs=None):
    '''
    Returns the irfs for all engoenous varibales given all possible AR1 shocks
    '''
    # make a dict
    if not isinstance(exog_irf, dict):
        exog_irf = {Z: exog_irf for Z in G.keys()}

    # calcualte irfs
    all_irfs = {Z: decompose_single_shock_irfs(G[Z], int_block_jacs, exog_irf[Z], Z, dxs=dxs) for Z in G.keys()}

    return all_irfs


## --- Varience ---
def irf_varience(irf, h=None):
    '''
    Returns the varience of a process based on its IRF
    '''
    return (irf[:h]**2).sum(axis=0)

def all_irf_variences(irfs, h=None):
    '''
    Returns the variences for all irfs in the irfs dictionary
    '''
    # initialize
    vars = {}

    # loop, gets vars
    for X, irf in irfs.items():
        vars[X] = irf_varience(irf, h=h)

    return vars

def all_shock_variences(irfs, h=None):
    '''
    Get all the variences for all the shocks
    '''
    # initializw
    int_vars = {}

    # loop over variables
    for Z, Xirfs in irfs.items():
        int_vars[Z] = all_irf_variences(Xirfs, h=h)
    
    # make it point X -> Z -> varience
    vars = {}
    for Z, Xvar in int_vars.items():
        for X, var in Xvar.items():
            if X not in vars:
                vars[X] = {}
            vars[X][Z] = var
    
    return vars


## --- Simulation ---
def sim_shock_path(G, shock_paths, rhos, dxs=None):
    '''
    Simulate aggregates after a shock given paths for the shocks
    '''
    ## setup
    T = next(iter(next(iter(G.values())).values())).shape[0]
    T_obs = next(iter(shock_paths.values())).shape[0]

    ## get irfs
    irfs = all_shock_irfs(G, rhos, T, dxs=dxs)

    ## loop to get all the paths
    decomp_series_paths = {}
    series_paths = {}
    for Z, Xirf in irfs.items():
        decomp_series_paths[Z] = {}

        ## get the path for each series
        for X, irf in Xirf.items():
            all_dX_path = irf[..., None] * shock_paths[Z]  # T by Tobs matrix, each column is an irf to the shock at time T
            dX_path = 1 * all_dX_path[0]  # flatten
            for i in range(1, min(T_obs, T)):
                dX_path[i:] += all_dX_path[i, :-i]

            # store results
            decomp_series_paths[Z][X] = dX_path
            add_or_insert(series_paths, X, dX_path)

    return series_paths, decomp_series_paths


def decomp_sim_shock_path(G, shock_paths, rhos, int_block_jacs, dxs=None):
    '''
    Simulate aggregates after a shock given paths for the shocks and
    seperates it into different endogenous channels
    '''
    ## setup
    T = next(iter(next(iter(G.values())).values())).shape[0]
    T_obs = next(iter(shock_paths.values())).shape[0]

    ## decomposed irfs
    decomp_irfs = decompose_all_irfs(G, int_block_jacs, {Z: ar_shock_irf(rhos[Z], T, 1) for Z in G.keys()}, G.keys(), dxs=dxs)

    ## loop to get all the paths
    decomp_series_paths = {}
    series_paths = {}
    for Z, XXirf in decomp_irfs.items():
        ## get the path for each series
        for Xres, Xirf in XXirf.items():
            if Xres not in decomp_series_paths:
                decomp_series_paths[Xres] = {}
            for Xcause, irf in Xirf.items():
                # path
                all_dX_path = irf[..., None] * shock_paths[Z]  # T by Tobs matrix, each column is an irf to the shock at time T
                dX_path = 1 * all_dX_path[0]  # flatten
                for i in range(1, min(T_obs, T)):
                    dX_path[i:] += all_dX_path[i, :-i]

                # store results
                add_or_insert(decomp_series_paths[Xres], Xcause, dX_path)
                add_or_insert(series_paths, Xres, dX_path)

    return series_paths, decomp_series_paths
