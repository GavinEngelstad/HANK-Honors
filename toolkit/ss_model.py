'''
Parent class for HA models. Allows us to find steady states of the models.
'''


## load some packages
from toolkit.grids import get_ss_tran_mat_dist
from toolkit.hamodel import HAModel
from scipy.optimize import root
from scipy import sparse
import numpy as np
import numba as nb


## class
class SSModel(HAModel):
    '''
    Parent class for an HA model that can solve for a steady state.
    '''

    def __init__(self, N_a, a_min, a_max, N_z, rho_z, sigma_z):
        '''
        HA model with the functions to find the steady state
        '''
        ## create grid
        super().__init__(N_a, a_min, a_max, N_z, rho_z, sigma_z)


    ## --- Endogenous Grid Method ---
    # iterate backwards to find policy rules for hh blocks
    def egm(self, V_a, egm_args, egm_back, max_iter, tol):
        '''
        Does EGM iterations for an HA model. Assumes the function `egm_back` returns the
        value function derivative `V_a_t` as the frst argument and takes the value function
        derivative as the first argument, then `egm_args`
        '''
        for _ in range(max_iter):
            # get new consumption function
            (V_a, egm_res), last_V_a = egm_back(self, V_a, *egm_args), V_a

            # exit condition
            diff = np.abs(V_a - last_V_a).max()
            if diff < tol:
                break
        else:
            raise RuntimeError(f'Value Function Iteration: Iteration Count Exceeded. Curent Tolerance = {diff}')
        
        return V_a, egm_res
    

    ## --- Transition Matrix ---
    # model discretized household transitions between states on the exogenous and
    # endogenous grid
    @staticmethod
    @nb.njit
    def make_tran_mat_vectors(a_t, a_grid, z_tran_mat):
        '''
        Make the sparse matrix vectors for the transition matrix. Sparse matricies can't
        be njited, so we do as much as possible here to make it faster before passing it
        to scipy
        '''
        # setup
        N_a = a_grid.size
        N_z = z_tran_mat.shape[0]

        # figure out where values go
        idxs = np.maximum(np.minimum(np.searchsorted(a_grid, a_t, 'right'), N_a - 1), 1)  # upper of the two adjacent indicies to where g maps
        p_l = np.empty(a_t.shape)
        for i in nb.prange(idxs.shape[0]):
            for j in nb.prange(idxs.shape[1]):
                idx = idxs[i, j]
                p_l[i, j] = max((a_grid[idx] - a_t[i, j]) / (a_grid[idx] - a_grid[idx-1]), 0.)
        # # 1 -> it maps to the lower gridpoint, 0 -> it maps to the higher girdpoint, 0.5 -> its halfway between both
        p_h = 1 - p_l  # how close it is to the upper half

        # create transision matricies
        # map capital now - > capital later
        data = np.vstack((p_l, p_h))  # each row resepsends a different level of capital
        row = np.vstack((idxs-1, idxs))
        col = np.hstack((np.arange(N_a), np.arange(N_a)))[:, None] + np.zeros((1, N_z))

        # add exogenous probability shocks
        data = (data[:, :, None] * z_tran_mat.T[None, :, :]).ravel()
        row = (row[:, :, None] + N_a * np.arange(N_z)).ravel()
        col = (col[:, None, :] + N_a * np.arange(N_z)[None, :, None]).ravel()
        
        return data, row, col

    def make_tran_mat(self, a_t):
        '''
        Makes the transition matrix between endogenous states on `a_grid` decided by
        `a_t` and exogenous states following `z_tran_mat`.
        '''
        data, row, col = SSModel.make_tran_mat_vectors(a_t, self.a_grid, self.z_tran_mat)
        dim = self.a_grid.size * self.z_tran_mat.shape[0]
        tran_mat = sparse.csr_array((data, (row, col)), shape=(dim, dim))  # sparse array cant be jit compiled

        return tran_mat
    

    ## --- Distribution ---
    # gets the steady state distribution of household states from the transition matrix
    def get_ss_dist(self, a_t):
        '''
        Gets the steady state distribution of endogenous and exogenous household states
        '''
        # get the transition matrix and distirbution
        tran_mat = self.make_tran_mat(a_t)
        dist = get_ss_tran_mat_dist(tran_mat)

        # reshape
        dist = dist.reshape((-1, self.z_tran_mat.shape[0]), order='F')

        return dist, tran_mat
    

    ## --- HA Block ---
    # calculates the steady state of the HA block. Runs EGM iterations, finds the
    # distribution, and aggregates variables
    def HA_ss(self, egm_back, V_a0, egm_args, agg_dict, max_iter, tol):
        '''
        Calculates the steady state distribution given the egm_args for the economy.

        `mod_pars` should contain 'a_grid' representing the asset grid and 'z_tran_mat'
        representing the exogenous transition matrix.
        
        `egm_back` should take `(V_a0, *egm_args)` as inputs and return a tuple
        `V_a, egm_res` where egm_res is a dictionary containing, amoung other outputs,
        'a' for the savings rule
        '''
        # egm
        V_a, egm_res = self.egm(V_a0, egm_args, egm_back, max_iter, tol)  # egm_res should contain 'a' for the savings rule
        self.V_a = V_a
        for x, val in egm_res.items():
            self.__setattr__(x, val)

        # get distribution 
        dist, tran_mat = self.get_ss_dist(self.a)
        self.dist = dist
        self.tran_mat = tran_mat

        # aggregate
        for X, (aggf, x) in agg_dict.items():
            self.__setattr__(X, aggf(self, self.__getattribute__(x)))


    ## --- Solve Steady State ---
    # solve the steady state of the model!
    def solve_ss(self, free, endog, V_a0, X0, simple_blocks=None, ha_block=None, markets=None, ss_tol=1e-12, egm_max_iter=10000, egm_tol=1e-14):
        '''
        Solves for the steady state of an HA model.

        Needs to be told what variables to search for (`free`), what to determine within the
        model (`endog`, should be ordered so that it can be determined), initial parameters
        for the model, the simple_blocks, ha_blocks, and markets of the model (sometimes you
        should restrict the markets from the full market clearing condtion if variables can
        be determined analytically), and intial guesses for the value function and free 
        parameters.

        Depending on the ititial guess, it may not converge and throw an error. If that happens,
        change the guess and try again
        '''
        # setup
        if simple_blocks is None:  # get these (if we use the one from the model)
            simple_blocks = self.simple_blocks
        if ha_block is None:
            ha_block = self.ha_block
        if markets is None:
            markets = self.markets
        self.V_a = V_a0  # reuse last iterations V_a in the loop

        def check_ss(X):
            # fill in ss_pars with the guess
            for v, g in zip(free, X):
                self.__setattr__(v, g)
            
            # calculate free in order of array if in simple blocks, assume the rest are in HA block (which is done last)
            for v in endog:
                if v in simple_blocks:
                    f, pars = simple_blocks[v]
                    self.__setattr__(v, f(self, *[self.__getattribute__(v2) for v2, _ in pars]))
        
            # ha block, calc after all the simple ones are done
            self.HA_ss(
                    egm_back = ha_block['egm'],
                    V_a0 = self.V_a,
                    egm_args = [self.__getattribute__(v) for v in ha_block['inputs']],  # inputs
                    agg_dict = ha_block['aggs'],  # aggregators
                    max_iter = egm_max_iter,
                    tol = egm_tol,
                )

            # market clearing
            mkts = np.array([f(self, *[self.__getattribute__(v2) for v2, _ in pars]) for f, pars in markets.values()])

            return mkts
        
        # solve for the root
        res = root(check_ss, X0, tol=ss_tol)
        assert res.success

        # flag
        self.ss = True
