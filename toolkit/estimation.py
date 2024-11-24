'''
Functions for Bayesian Estimation of HA models
'''

## load some packages
from scipy.optimize import minimize
from scipy import linalg
import numba as nb
import numpy as np

## --- Priors ---
def log_inv_gamma(X, mu, sig):
    ## evaluates in a log scale, makes probabilities additive
    alpha = (mu / sig)**2 + 2
    beta = mu * (alpha - 1)
    return -(alpha + 1) * np.log(X) - beta / X
    # actual equation is alpha log beta - gamma(alpha) - (alpha - 1) log X - beta / X
    # we drop the constant since it doesn't affect comparisons


def log_beta(X, mu, sig):
    alpha = (mu * (1 - mu) - sig ** 2) / (sig ** 2 / mu)
    beta = alpha / mu - alpha
    return (alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X)
    # actual equation is (alpha - 1) log X + (beta - 1) log (1 - X) - log B(alpha, beta)
    # we drop the constant since it doesn't affect comparisons


## --- G Matrix ---
# for this, we work with a larger G matrix so all the irfs becomes one instance of matrix multiplication
# here, we make that matrix
def make_shock_Gmat(G, outputs=None):
    '''
    Makes the G matrix thats plugged into estimations for one shock

    G should be for one shock
    '''
    # make ooutputs list
    if outputs is None:
        outputs = G.keys()
    
    # make the matrix
    Gmat = np.stack([G[X] for X in outputs])

    return Gmat


def estimation_G(G, outputs=None):
    '''
    Makes the G matrix thats plugged into estimations
    '''
    return {Z: make_shock_Gmat(G[Z], outputs) for Z in G.keys()}


## --- Covariences ---
def ar_exog_irf(rho, T):
    '''
    Returns the irf for an ar1 shock
    '''
    return rho**np.arange(T)


def shock_irfs(G, rho):
    '''
    Returns the irf for a single shock
    '''
    T = G.shape[-1]

    # get the exogenous irf
    exog_irf = ar_exog_irf(rho, T)

    # make the irfs
    irfs = G @ exog_irf

    return irfs


def all_shock_irfs(Gs, rhos):
    '''
    Returns all the shock irfs for a matrix G
    '''
    ## for each we
    # 1. get the exognous irf
    # 2. create the irf for endogneous variables in G
    return np.stack([shock_irfs(Gs[Z], rhos[i]) for i, Z in enumerate(Gs.keys())])


def covariences(irfs, sigmas):
    '''
    Returns the covariences between varaibles based on the irfs for them
    '''
    T = irfs.shape[-1]

    # this is lowkey coppied from auclert
    dft = np.fft.rfftn(irfs.T, (2 * T - 2,), axes=(0,))  # fft (in any number of dimensions with stacked 0s)
    total = (dft.conjugate() * sigmas**2) @ dft.swapaxes(1, 2)  # inside product
    cov = np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]  # inverse fft (in any number of dimensions with stacked 0s)

    return cov


## --- Log liklihood ---
@nb.njit  # this is slow, needs to be jitted
def stack_covs(cov, T_obs, meas_err):  # works with numba/numpy, jax jit is finicky and i couldnt make it work (well)
    '''
    Stack the covariences into something we can evaluate the liklihood of
    '''
    # config
    T, N_x, _ = cov.shape
    V = np.zeros((T_obs * N_x, T_obs * N_x))

    # loop over options
    for t1 in range(T_obs):  # leave at 0 if outside t range, outside matrix if at T_obs range
        for t2 in range(max(0, t1 - T + 1), min(T_obs, t1 + T)):
            if t1 == t2:
                cov_t1_t2 = (np.diag(meas_err**2) + cov[0] + cov[0].T) / 2  # auclert says to do this to keep it symetric, i dont htink it matters
            elif t1 < t2:
                cov_t1_t2 = cov[t2 - t1]  # covarience of t1 with repect to t2
            else:
                cov_t1_t2 = cov[t1 - t2].T  # flip it, we get covarience of t2 with reprect to t1
            V[t1 * N_x: (t1 + 1) * N_x, t2 * N_x: (t2 + 1) * N_x] = cov_t1_t2
    
    return V


def log_liklihood(rhos, sigmas, G, data, meas_err=None, T_max=None):
    '''
    Evaluates the log-likihood of a model against data 
    '''
    # data
    T_obs, Nx = data.shape
    data = data.ravel()
    if meas_err is None:
        meas_err = np.zeros(Nx)

    # get the impulse response functions and standard deviations (used in covarience calcualtion)
    irfs = all_shock_irfs(G, rhos)

    # get covarience
    cov = covariences(irfs, sigmas)[:T_max]  # filter to first Tmax periods

    # make V
    V = stack_covs(cov, T_obs, meas_err)

    # log liklihood formula
    # want to find -(log det V + data * V^(-1) * data) / 2
    V_factored = linalg.cho_factor(V)
    d_Vinv_d = data @ linalg.cho_solve(V_factored, data)
    log_det_V = 2 * np.sum(np.log(np.diag(V_factored[0])))
    log_l = - (d_Vinv_d + log_det_V) / 2

    return log_l


def gen_posterior_prob(G, data, priors, m=1, meas_err=None, T_max=None):
    '''
    Generates a function that evaluates the log liklihood function and evaluates it at
    a parameter value 
    '''
    def posterior_prob(X):
        # get model log liklihood
        n = len(X) // 2
        rhos = X[:n]  # peristsance is first half of X, sigmas is second half
        sigmas = X[n:]
        logl = log_liklihood(rhos, sigmas, G, data, meas_err=meas_err, T_max=T_max)

        # prior probability
        for i, x in enumerate(X):
            logl += priors[i](x)
        print(X, logl)

        return logl * m  # m lets you minimize too
    
    return posterior_prob


## --- Mode ---
def posterior_mode(drawf, N_x, X0=None, bounds=None, tol=1e-12):
    '''
    Evaluate the posterior mode for estimation
    '''
    ## setup
    if X0 is None:  # initial guess
        X0 = 0.5 * np.ones(N_x * 2)
    if bounds is None:
        bounds = [(0 + tol, 1 - tol) for _ in range(N_x)] + [(0 + tol, None) for _ in range(N_x)]

    ## find the minimum
    res = minimize(
            lambda x: -drawf(x),
            X0,
            bounds=bounds
        )

    return res


## --- Hessian ---
def get_invhessian(f, X, h=1e-4):
    '''
    Computes the inverse hessian for a function at X using numeric differentiation. Used
    at the mode of the liklihood function to generate the cov for the nultivariate normal
    draws in the RWMH algorithm
    '''
    # setup
    N = len(X)
    Ih = h * np.eye(N)  # identity times h

    # make the hessian
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            ff = f(X + Ih[i] + Ih[j])  # forwards for both
            fb = f(X + Ih[i] - Ih[j])  # forwards, backwards
            bf = f(X - Ih[i] + Ih[j])  # backwards, forward
            bb = f(X - Ih[i] - Ih[j])  # backwards for both
            H[i,j] = -(ff - fb - bf + bb) / 4 / h / h  # - makes sure its positive definite
    H += H.T  # hessian is symetric

    # inverse
    invH = np.linalg.inv(H)

    return invH


## --- Metropolis Hastings ---
def metropolis_hastings(drawf, X0, sigmas, bounds, N_sim, N_burn):
    # initialize vectors
    sim_res = np.empty((N_sim, X0.shape[0]))
    logposterior = np.empty(N_sim)
    X = X0
    accept = 0

    # inital evalupation
    obj = drawf(X)

    # iterate forwad
    for i in range(N_sim - 1):
        # log last period
        sim_res[i] = X
        logposterior[i] = obj

        # jump
        X_proposed = np.random.multivariate_normal(X, sigmas)

        # check bounds, if bounds check fails we reject
        if (X_proposed <= bounds[0]).any():
            continue
        if (X_proposed >= bounds[1]).any():
            continue
        
        # evaluate liklihood
        obj_proposed = drawf(X_proposed)
        p_accept = np.exp(np.min((0, obj_proposed - obj)))  # avoid overflow errors
        # if obj_proposed > obj, then e^(obj_proposed - obj) > e^(0) = 1, so we fs accept, therefore we ignore that condition
        if np.random.rand() <= p_accept:  # accept
            X = X_proposed
            obj = obj_proposed
            accept += 1
    # log final one
    sim_res[-1] = X
    logposterior[-1] = obj

    # burn
    sim_res = sim_res[N_burn:]
    logposterior = logposterior[N_burn:]

    # calulate acceptance rate
    accept_rate = accept / N_sim

    return sim_res, logposterior, accept_rate
