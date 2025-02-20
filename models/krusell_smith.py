'''
A simple Krusell Smith model with a TFP shocks. To solve it, I use the method
from Auclert et al. "Using the Sequence-Space Jacobian to Solve and Estimate
Heterogeneous-Agent Models."

The model features a Heterogenous households and a firm block.
'''

## load some packages
from toolkit.helpers import agg_dist, func_interp
from toolkit.shock_model import ShockModel
import numba as nb
import numpy as np


# KrusellSmith class, contains code to solve and run a krusell smith
class KrusellSmith(ShockModel):
    '''
    Krusell Smith baseline model to test steady state and shock code
    '''

    ## --- Initialization ---
    # Setup the grids and store variables within the object
    def __init__(self, N_a, a_min, a_max, N_z, rho_z, sigma_z, **kwargs):
        '''
        Initializer for the Krussell Smith model.

        `N_a`, `a_min`, `a_max`, `N_z`, `rho_z`, and `sigma_z` are passed to super to
        create the grids

        kwargs should contain the model parameters. Everything thats not either calibrated
        or determined endogenously should be in this.
        
        It needs to include (otherwise calibration is impossible):
         - Shock Steady States: A

        It can include (if they're not targets in calibration):
         - Household Parameters: beta, gamma
         - Firm Parameters: delta, alpha
         - Targets: Y
        '''
        # save variables as attributes, these are used in each of the functions
        for x, val in kwargs.items():
            self.__setattr__(x, val)


        # create grid
        super().__init__(N_a, a_min, a_max, N_z, rho_z, sigma_z)

        # analytic
        self.L = agg_dist(self.z_dist, self.z_grid)


    ## --- Model Setup ---
    # Charcterization of the model

    ## Non-HA Characterization
    # for each equation, there is an `input` list. This list includes the variables
    # in the model in order and their shift amount. A positive shift implies past
    # periods values are used in this periods equation, since it has to be shifted
    # forward then used in the equation
    # Functions take the values in order and the model params as inputs

    # firm block
    # Takes: K
    # Gives: Y, W, R
    inputY = [('K', 1), ('A', 0)]
    def fY(self, K_l, A_t):
        return A_t * K_l**self.alpha * self.L**(1 - self.alpha)

    inputW = [('K', 1), ('A', 0)]
    def fW(self, K_l, A_t):
        return (1 - self.alpha) * A_t * K_l**self.alpha * self.L**(-self.alpha)

    inputR = [('K', 1), ('A', 0)]
    def fR(self, K_l, A_t):
        return self.alpha * A_t * K_l**(self.alpha - 1) * self.L**(1 - self.alpha) + 1 - self.delta
    
    # govt block
    # takes: g, Y
    # gives: G, T
    inputG = [('g', 0), ('Y', 0)]
    def fG(self, g_t, Y_t):
        return g_t * Y_t
    
    inputtau = [('G', 0)]
    def ftau(self, G_t):
        return G_t

    ## --- Market clearing ---
    # Takes: K, curlK
    # Returns: 0 (hopefully) for capital clearing
    inputmkt = [('K', 0), ('curlK', 0)]
    def fmkt(self, K_t, curlK_t):
        return K_t - curlK_t
    
    ## --- Household block ---
    # This is the heterogenous part, so it's a little more complex. We use the
    # endeogenous grid method to solve for policy rules, follow Young to travel
    # between points
    @staticmethod
    @nb.njit
    def egm_back_jit(V_a_p, W_t, R_t, T_t, z_grid, beta, z_tran_mat, gamma, a_grid, N_z):
        # setup
        Wz_t_grid = W_t * z_grid  # wage at each gridpoint

        # egm steps (all on next periods gridpoints)
        dc_t_nextgrid = V_a_p @ (beta * z_tran_mat)
        c_t_nextgrid = dc_t_nextgrid**(-1 / gamma)
        Ra_l_nextgrid = a_grid[:, None] + c_t_nextgrid - Wz_t_grid + T_t

        # convert to this periods gridpoints
        a_t = func_interp(R_t * a_grid, Ra_l_nextgrid, a_grid.repeat(N_z).reshape((-1, N_z)))  # by linearity can multiply lhs by R instead of div rhs
        c_t = Wz_t_grid + R_t * a_grid[:, None] - a_t - T_t
        V_a_t = R_t * c_t**(-gamma)

        return V_a_t, a_t, c_t

    def egm_back(self, V_a_p, W_t, R_t, T_t):
        V_a_t, a_t, c_t = KrusellSmith.egm_back_jit(
                V_a_p, W_t, R_t, T_t,
                self.z_grid, self.beta, self.z_tran_mat, self.gamma, self.a_grid, self.N_z
            )

        return V_a_t, {'a': a_t, 'c': c_t}
    
    def agg_dist(self, x):
        return agg_dist(self.dist, x)

    ## config
    # config dictionaries
    # simple_blocks: dict points variable to evaluation function and inputs
    simple_blocks = {  # order matters, needs to be in an order that can be evaluated
            'Y': (fY, inputY), 'W': (fW, inputW), 'R': (fR, inputR),  # firm block
            'G': (fG, inputG), 'tau': (ftau, inputtau),  # govt block
        }
    # ha_block: only one ha block, so in an order list. Has egm function, inputs, aggregators
    ha_block = {  # egm func
        'egm': egm_back,
        'inputs': ['W', 'R', 'tau'], 
        'aggs': {
            'curlK': (agg_dist, 'a'),
            'curlC': (agg_dist, 'c'),
        },
    }
    # markets: dict points market to evaluation function (goes to 0) and inputs
    markets = {
            'mkt': (fmkt, inputmkt),  # capital market
        }
    vars = ['K']  # passed into the function
    shocks = ['A', 'g']
    exog = vars + shocks


if __name__ == '__main__':
    # imports
    from toolkit.irfs import single_shock_irfs, decompose_single_shock_irfs, policy_irf, decompose_policy_irf
    import matplotlib.pyplot as plt


    ## --- Solve Steady State ---
    # parameterization
    mod_pars = {
            'beta': 0.95, 'gamma': 2., 'delta': 0.05,  # household parameters
            'N_z': 7, 'rho_z': 0.9, 'sigma_z': 0.1,  # household state transitions
            'alpha': 0.3,  # govt parameters
            'A': 1.,  # shock steady state
            'N_a': 501, 'a_min': 0, 'a_max': 500,  # grid states
        }

    # ks object, everything is done by this!
    ks = KrusellSmith(**mod_pars)

    # calibration parameters
    free = ['K']
    endog = ['Y', 'W', 'R']
    V_a0 = (ks.a_grid[:, None] * (1 / ks.beta - 1) + ks.z_grid)**(-ks.gamma)
    X0 = np.array([5.])

    # calibrate the hank
    ks.solve_ss(free, endog, V_a0, X0)
    assert np.isclose(ks.curlC + ks.K * ks.delta - ks.Y, 0)  # walras

    ## plot steady state policy functions
    # setup plot
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(10)
    n = (ks.a_grid <= 10).sum()  # number of gridpoints we plot
    axs[0].set_title('Savings Rule')
    axs[0].set_xlabel('$a_{i, t-1}$')
    axs[0].set_ylabel('$a_{i, t}$')
    axs[1].set_title('Consumption Rule')
    axs[1].set_xlabel('$a_{i, t-1}$')
    axs[1].set_ylabel('$c_{i, t}$')

    # savings rule
    axs[0].plot(ks.a_grid[:n], ks.a[:n], label=['$z_{i, t}='+str(round(z, 2))+'$' for z in ks.z_grid])
    axs[0].plot([0, ks.a_grid[n - 1]], [0, ks.a_grid[n - 1]], 'k--', label='45 Degree Line')

    # consumoption rule
    axs[1].plot(ks.a_grid[:n], ks.c[:n])

    # final things
    fig.tight_layout()
    fig.subplots_adjust(right=8/10)
    fig.legend(loc='center left', bbox_to_anchor=(8/10, 0.5), frameon=False)


    ## --- Shock the Model ---
    # solve for the shock response matricies 
    T = 300  # time horizen for shocks
    G, dxs = ks.solve_G(T, ha=True)  # if you just want normal irfs, run with ha=False

    # find the irf
    irfs, exog_irf = single_shock_irfs(G['A'], 0.9, T)  # set rho = 0.9
    decomp = decompose_single_shock_irfs(G['A'], ks.int_block_jacs, exog_irf, 'A')  # cool plots

    # plot config
    fig, axs = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.suptitle('1% TFP Shock')

    # plot irfs
    vars = ['K', 'Y', 'W', 'R', 'curlK', 'curlC']  # variables we're interested in
    labels = {
            'K': '$K$', 'Y': '$Y$', 'W': '$W$', 'R': '$R$',  # aggs
            'curlK': r'$\mathcal{K}$', 'curlC': r'$\mathcal{C}$',  # het block
            'A': '$A$',  # shock
        }
    t = 100  # plot first 100 periods after shock
    for i in range(6):
        # variables for i
        v = vars[i]
        ax = axs.take(i)

        # plot irf
        ax.set_title(labels[v])
        ax.set_xlabel('$t$')
        ax.set_ylabel('% Dev. From SS')
        ax.plot(irfs[v][:t] / ks.__getattribute__(v), label='Aggregate Effects')

        # decomposition
        if v not in decomp.keys():
            continue
        for v2, irf in decomp[v].items():
            ax.plot(irf[:t], '--', label=f'{labels[v2]} Effects')

        # legend
        ax.legend()

    # final things
    fig.tight_layout()


    ## --- Household Policy Rule Shocks ---
    # config
    ts = [0, 25, 50]
    n = (ks.a_grid < 10).sum()  # number of gridpoints we plot

    # solve hh impulse response
    pol_irf = policy_irf(dxs['c'], G['A'], exog_irf, 'A')
    pol_decomp = decompose_policy_irf(dxs['c'], G['A'], exog_irf, 'A')

    # plot config
    fig, axs = plt.subplots(3, 3, sharey='row')
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.suptitle('1% TFP Shock, HH $c$ Response')
    fig.supxlabel('$a$')
    fig.supylabel('% Dev From SS')

    # plot irfs
    for i in range(3):
        # variables for i
        t = ts[i]
        # ax = axs.take(i)

        # plot irf
        axs[0, i].set_title(f'$t = {t}$')
        lns = axs[0, i].plot(ks.a_grid[:n], pol_irf[t, :n] / ks.c[:n], label=['$z='+str(round(z, 2))+'$' for z in ks.z_grid])

        # decomposition
        axs[1, i].set_title('$W$ Effects')
        axs[1, i].plot(ks.a_grid[:n], pol_decomp['W'][t, :n] / ks.c[:n])
        axs[2, i].set_title('$R$ Effects')
        axs[2, i].plot(ks.a_grid[:n], pol_decomp['R'][t, :n] / ks.c[:n])

    # final things
    fig.tight_layout()
    fig.subplots_adjust(right=8/10)
    fig.legend(loc='center left', bbox_to_anchor=(8/10, 0.5), frameon=False, handles=lns)

    # show the plots
    plt.show()
