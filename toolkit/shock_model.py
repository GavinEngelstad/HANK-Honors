'''
Parent class for ha models. Allows you to shock the model
'''

# load some packages
from toolkit.simple_sparse import SimpleSparse
from toolkit.helpers import add_or_insert
from toolkit.ss_model import SSModel
import numpy as np
import numba as nb
import jax
jax.config.update('jax_enable_x64', True)


## class
class ShockModel(SSModel):
    '''
    Parent class for an HA model that can have shocks applied to it
    '''

    def __init__(self, N_a, a_min, a_max, N_z, rho_z, sigma_z):
        '''
        HA model with the functions to find the steady state
        '''
        ## create grid
        super().__init__(N_a, a_min, a_max, N_z, rho_z, sigma_z)

    
    ## --- Simple Block ---
    # find simple block jacobans
    # these are all represented efficently as SimpleSparse matricies
    def get_simple_jac(self, eqn):
        '''
        Gets the SimpleSparse jacobians of a simple block equation (or market clearing
        condition). The equation should be in either the `simple_blocks` list or
        `markets` list
        '''
        if eqn in self.simple_blocks:
            f, inputs = self.simple_blocks[eqn]
        else:  # in markets
            f, inputs = self.markets[eqn]
        
        # loop over inputs
        jacs = {}
        ss_inpt = [self.__getattribute__(v) for v, _ in inputs]
        for i, (v, s) in enumerate(inputs):
            ss_jac = SimpleSparse([s], [0], [jax.grad(f, 1 + i)(self, *ss_inpt)])  # 1 + bc self is first argument
            add_or_insert(jacs, v, ss_jac)
        
        return jacs
    
    def collect_simple_jacs(self, final_jacs_dict, int_jacs_dict, eqns):
        '''
        Collects the jacobains for all `eqns` into `jacs_dict`.

        `eqns` should be in an order that can be evaluated to model input level
        '''
        for eqn in eqns:
            int_jacs_dict[eqn] = {}  # initialize the dicitonary for this variable 
            final_jacs_dict[eqn] = {}
            bl_jacs = self.get_simple_jac(eqn)  # get the 'simple' jacobians, not nessisarily of exogenous values
            for v, jac in bl_jacs.items():
                add_or_insert(int_jacs_dict[eqn], v, jac)
                if v in self.exog:  # exogenous, add directly to dictionary
                    add_or_insert(final_jacs_dict[eqn], v, jac)
                else:  # endogenous, use the jacobains for it and chian rule to make exogenous
                    for ex, jacv in self.block_jacs[v].items():
                        add_or_insert(final_jacs_dict[eqn], ex, jac @ jacv)

        return final_jacs_dict, int_jacs_dict
    

    ## --- Heterogenous Block ---
    # two-sided diff with the fake news algorithm to find the jacobians of ha block
    def collect_back_iter(self, s, V_a_pos, egm_res_pos, V_a_neg, egm_res_neg, curlYs, curlD, dxs, agg_dict, h):
        '''
        Collect the results from a back iteration in the fake news algorithm
        '''
        # dV_a, div by 2 so we can just add it on both sides in the next iter
        dV_a = (V_a_pos - V_a_neg) / 2

        # outputs
        for X, (aggf, x) in agg_dict.items():  # aggregate
            dx = (egm_res_pos[x] - egm_res_neg[x]) / h
            if dxs is not None:
                dxs[x][s] = dx
            curlYs[X][s] = aggf(self, dx)
        da = (egm_res_pos['a'] - egm_res_neg['a']) / 2

        # distirbution changes
        dtran_mat = self.make_tran_mat(self.a + da) - self.make_tran_mat(self.a - da)  # change in the transition matrix
        curlD[s] = (dtran_mat @ self.dist.ravel('F')).reshape((-1, self.N_z), order='F') / h  # change in the distribution, linearity

        return dV_a

    def back_iter(self, var, T, h, ha):  # ha = True if you want ha block policy functions too
        '''
        Baskwards iterates to find the distribution and output effects at each time

        Part of the fake news algorithm
        '''
        # setup
        egm_back = self.ha_block['egm']
        egm_args = self.ha_block['inputs']
        agg_dict = self.ha_block['aggs']
        inputs = np.array([self.__getattribute__(v) for v in egm_args])
        dinputs = h / 2 * (np.array(egm_args) == var)  # h / 2 for the input, 0 elsewhere, two sided diff so do h/2
        curlYs = {X: np.empty(T) for X in agg_dict.keys()}  # curlYs[X][s] represents effect on aggregate x at time 0 of shock at time s
        if ha:
            dxs = {x: np.empty((T, self.N_a, self.N_z)) for _, x in agg_dict.values()}  # dxs[x][s] = effect on policy rule x at time 0 of shock at time s
        else:
            dxs = None
        curlD = np.empty((T, self.N_a, self.N_z))  # curlD[s] represents effect on period 1 distribution of shock at time s

        # intial period, perturbed input
        V_a_pos, egm_res_pos = egm_back(self, self.V_a, *(inputs + dinputs))  # positive and negative diff
        V_a_neg, egm_res_neg = egm_back(self, self.V_a, *(inputs - dinputs))
        dV_a = self.collect_back_iter(0, V_a_pos, egm_res_pos, V_a_neg, egm_res_neg, curlYs, curlD, dxs, agg_dict, h)

        # get later periods, perturbed V_a
        for s in range(1, T):
            V_a_pos, egm_res_pos = egm_back(self, self.V_a + dV_a, *inputs)  # positive and negative diff
            V_a_neg, egm_res_neg = egm_back(self, self.V_a - dV_a, *inputs)
            dV_a = self.collect_back_iter(s, V_a_pos, egm_res_pos, V_a_neg, egm_res_neg, curlYs, curlD, dxs, agg_dict, h)

        return curlYs, curlD, dxs
    
    def expect_vectors(self, xvar, T):
        '''
        Find expectation vectors
        '''
        # initialize
        x = self.__getattribute__(xvar)
        curly_E = np.empty((T-1, self.N_a, self.N_z))  # curly_E[s] represents expected x at time s

        # expect = transition matrix * wher eyou exepcetd to be last period
        curly_E[0] = x  # how much they decide to spend (known now)
        xvec = x.ravel('F')
        for t in range(1, T-1):
            xvec = self.tran_mat.T @ xvec
            curly_E[t] = xvec.reshape((-1, self.N_z), order='F')

        return curly_E

    @staticmethod
    @nb.njit
    def fake_news(curlY, curlD, curlE, T):
        '''
        Make the fake news matrix from the curlys
        '''
        # initialize and assign top row
        fake_news_mat = np.empty((T, T))
        fake_news_mat[0] = curlY

        # make bottom rows
        for s in range(T-1):
            for t in range(T):
                fake_news_mat[s+1, t] = (curlE[s] * curlD[t]).sum()

        return fake_news_mat
    
    @staticmethod
    @nb.njit
    def jacobian_from_fake_news(fake_news_mat):
        '''
        Make the jacobian from the fake news matrix
        '''
        # initialize it with the fake news matrix, we'll add elemtns to iteself to make the jacobians
        jacobian = 1. * fake_news_mat

        # add down the diaganel
        for i in range(1, jacobian.shape[0]):
            jacobian[i, 1:] += jacobian[i-1, :-1]
        
        return jacobian
    
    def collect_ha_jacs(self, final_jacs_dict, int_jacs_dict, T, h, ha):  # ha = True if you want household policy rule shocks
        '''
        Gets the heterogenous block jacobians
        '''
        # setup
        agg_dict = self.ha_block['aggs']
        inputs = self.ha_block['inputs']
        aggs = list(agg_dict.keys())  # idk if this needs to be a list tbh
        policies = [x for _, x in agg_dict.values()]  # household policy rules
        self.fake_news_mats = {X: {} for X in aggs}
        for X in aggs:
            final_jacs_dict[X] = {}  # create empty array
            int_jacs_dict[X] = {}

        # expectation vectors
        curlEs = {x: self.expect_vectors(x, T) for x in policies}

        # for each input, get the jacobian
        if ha:
            self.dxs = {x: {} for x in policies}
        for inp in inputs:
            curlYs, curlD, dxs = self.back_iter(inp, T, h, ha)  # get change in Y, D
            if ha:
                for x, dx in dxs.items():
                    self.dxs[x][inp] = dx
            for X, curlY in curlYs.items():
                x = agg_dict[X][1]  # policy rule

                # get fake news matrix
                fake_news_mat = ShockModel.fake_news(curlY, curlD, curlEs[x], T)
                self.fake_news_mats[X][inp] = fake_news_mat

                # get jacobian
                jac = self.jacobian_from_fake_news(fake_news_mat)
                add_or_insert(int_jacs_dict[X], inp, jac)  # store intermediate jacobians
                if inp in self.exog:  # exogenous, add directly to dictionary
                    add_or_insert(final_jacs_dict[X], inp, jac.copy())
                else:  # endogenous, use the jacobains for it and chian rule to make exogenous
                    for ex, jacv in self.block_jacs[inp].items():
                        add_or_insert(final_jacs_dict[X], ex, jac @ jacv)
        
        return final_jacs_dict, int_jacs_dict


    ## --- Solve for G ---
    # find the G matrix to get the model response to shocks
    def make_jac_F(self, vars, T):
        ## takes the jacobains for the markets and a list of varaibles and makes the jacobain
        # for f with repoect to those varaibles
        J_X = []  # keep blocks here
        for x in vars:
            J_x = []  # keep variable specific jacobians here
            for jacs in self.market_jacs.values():
                if x in jacs:
                    jac = jacs[x]
                    if isinstance(jac, SimpleSparse):
                        jac = jac.to_numpy(T)  # turn jacobain into a numpy array to invert
                    J_x.append(jac.T)
                else:
                    J_x.append(np.zeros((T, T)))  # fill in zeros if no jabobian exists
            J_X.append(J_x)  # add list to the overall thing
        J_X = np.block(J_X).T  # transpose of blcoks is the jacobian
        
        return J_X

    def solve_G(self, T, h=1e-6, ha=False):
        '''
        Solves for the G matrix to get irfs for the model
        '''
        # make sure we have a steady state
        if not self.ss:
            raise ValueError('Find a Steady State First')

        # save variables
        self.T = T
        self.block_jacs = {}  # this will be filled in by the simple blocks and ha jacobains with respect to exog
        self.int_block_jacs = {}  # this will be filled in by the simple blocks and ha jacbobains with respect to direct inputts

        # solve for the simple block jacobians
        self.collect_simple_jacs(self.block_jacs, self.int_block_jacs, self.simple_blocks.keys())  # fills in the needed parts of block jacs as it goes

        # solve for the heterogenous block jacobians
        self.collect_ha_jacs(self.block_jacs, self.int_block_jacs, T,  h, ha)

        # get market clearing jacobians
        self.market_jacs, _ = self.collect_simple_jacs({}, {}, self.markets.keys())

        # collect, make G
        J_Z = self.make_jac_F(self.shocks, T)
        J_X = self.make_jac_F(self.vars, T)
        Gmat = -np.linalg.solve(J_X, J_Z)

        # get variable by variable Gs
        # get individal variable matricies
        self.Gmat = {}
        for i, s in enumerate(self.shocks):  # want G for endgoenous variables with respect to each of the shocks
            Gs = {}  # G for shock s

            # variables in F
            for j, v in enumerate(self.vars):
                Gs[v] = Gmat[j * T: (j + 1) * T, i * T: (i + 1) * T]
            
            # other endogenous variables
            for v, jacs in self.block_jacs.items():
                # direct exogenous effect
                if s in jacs:
                    Gs[v] = jacs[s].copy()
                
                # indirect effect from endog
                for v2, jac in jacs.items():
                    if v2 in self.shocks:
                        continue
                    add_or_insert(Gs, v, jac @ Gs[v2])
            
            self.Gmat[s] = Gs

        if ha:
            return self.Gmat, self.dxs
        return self.Gmat
