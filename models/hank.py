'''
A 1 asset, discrete time HANK model wtih 6 shocks applies. The model is outlined
in the paper. To solve it, I use the method from the Auclert et al. "Using the
Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models."

The model features a Heterogenous households and a block for firms, monetary policy
and the government.
'''

## load some packages
from toolkit.helpers import agg_dist, func_interp
from toolkit.shock_model import ShockModel
import numpy as np
import numba as nb


# HANK class, contains code to solve and run a hank
class HANK(ShockModel):
    '''
    HANK model used in the paper
    '''

    ## --- Initialization ---
    # Setup the grids and store variables within the object
    def __init__(self, N_a, a_min, a_max, N_z, rho_z, sigma_z, **kwargs):
        '''
        Initializer for the HANK model.

        `N_a`, `a_min`, `a_max`, `N_z`, `rho_z`, and `sigma_z` are passed to super to
        create the grids

        kwargs should contain the model parameters. Everything thats not either calibrated
        or determined endogenously should be in this.
        
        It needs to include (otherwise
        calibration is impossible):
         - Philips Curve: kappa
         - Govt Parameters: rho_B, omega_pi, omega_Y
         - Shock Steady States: A, psi, g, xi, tauP, eta
         - Variable Steady States: B, pi (govt targets in model)

        It can include (if they're not targets in calibration):
         - Household Parameters: beta, gamma, chi, phi
         - Targets: I, Y, etc
        '''
        # save variables as attributes, these are used in each of the functions
        for x, val in kwargs.items():
            self.__setattr__(x, val)


        # create grid
        super().__init__(N_a, a_min, a_max, N_z, rho_z, sigma_z)

        # can find analytically
        self.W = 1 / self.psi
    

    ## --- Model Setup ---
    # Config for the model
    # Here, I include the model characterization (for HA and non-HA blocks),
    # and the order for the model to be solved
    # Generally, the model takes in output, wages, inflation, bonds as well as
    # the shocks, and attempts to clear the philips cuve, bond law of motion,
    # labor market, and bond market


    ## --- Non-HA Characterization ---
    # for each equation, there is an `input` list. This list includes the variables
    # in the model in order and their shift amount. A positive shift implies past
    # periods values are used in this periods equation, since it has to be shifted
    # forward then used in the equation
    # Functions take the values in order and the model params as inputs

    # monetary policy block
    # Takes: pi, Y, xi
    # Returns: R, I
    inputI = [('pi', 0), ('Y', 0), ('xi', 0)]
    def fI(self, pi_t, Y_t, xi_t):
        return self.I * (pi_t / self.pi)**self.omega_pi * (Y_t / self.Y)**self.omega_Y * xi_t

    inputR = [('I', 1), ('pi', 0)]
    def fR(self, I_l, pi):
        return I_l / pi

    # govt block
    # Takes: g, Y, R, B, eta, tauP
    # Returns: G, tauL
    inputG = [('g', 0), ('Y', 0)]
    def fG(self, g_t, Y_t):
        return g_t * Y_t

    inputtauL = [('R', 0), ('B', 1), ('G', 0), ('B', 0), ('eta', 0), ('tauP', 0)]
    def ftauL(self, R_t, B_l, G_t, B_t, eta_t, tauP_t):
        T = agg_dist(self.z_dist, self.z_grid**tauP_t)  # tax progressivity
        return (R_t * B_l + G_t + eta_t - B_t) / T

    # firm block
    # Takes: Y, A, pi, W
    # Returns: N, M, D
    inputN = [('Y', 0), ('A', 0)]
    def fN(self, Y_t, A_t):
        return Y_t / A_t

    inputM = [('pi', 0), ('Y', 0)]
    def fM(self, pi_t, Y_t):  # M, FOC is 0 so we can prolly remove it?
        return self.kappa / 2 * (pi_t / self.pi - 1)**2 * Y_t

    inputD = [('Y', 0), ('W', 0), ('N', 0), ('M', 0)]
    def fD(self, Y_t, W_t, N_t, M_t):
        return Y_t - W_t * N_t - M_t

    ## --- Market Clearing ---
    # Essnetually just acts like a simple block
    # Takes: Y, W, pi, psi, A, R, G, eta, B, N, curlN, curlB  (curlX is X demand from HA block)
    # Returns: 0, 0, 0, 0 (hopefully) for philips curve, bond law of motion, labor market, and bond market
    inputmkt1 = [('pi', 0), ('pi', -1), ('psi', 0), ('W', 0), ('A', 0), ('R', -1), ('Y', -1), ('Y', 0)]
    def fmkt1(self, pi_t, pi_p, psi_t, W_t, A_t, R_p, Y_p, Y_t):
        hatpi_t = pi_t / self.pi - 1  # log deviation from steady state
        hatpi_p = pi_p / self.pi - 1
        # lhs = self.kappa * hatpi_t * (hatpi_t - 1)
        # rhs = 1 - psi_t + psi_t * W_t / A_t + self.kappa / R_p * Y_p / Y_t * hatpi_p * (hatpi_p - 1)
        return Y_p / Y_t / R_p * hatpi_p + self.kappa * (W_t / A_t - 1 / psi_t) - hatpi_t

    inputmkt2 = [('B', 0), ('R', 0), ('B', 1), ('G', 0), ('eta', 0)]
    def fmkt2(self, B_t, R_t, B_l, G_t, eta_t):
        RBdiff = R_t * B_l - self.R * self.B  # level devation from steady state
        Gdiff = G_t - self.G
        etadiff = eta_t - self.eta
        return B_t - self.B - self.rho_B * (RBdiff + Gdiff + etadiff)

    inputmkt3 = [('N', 0), ('curlN', 0)]
    def fmkt3(self, N_t, curlN_t):
        return N_t - curlN_t

    inputmkt4 = [('B', 0), ('curlB', 0)]
    def fmkt4(self, B_t, curlB_t):
        return B_t - curlB_t
    
    # inputmkt4 = [('Y', 0), ('curlC', 0), ('G', 0), ('M', 0)]  # agg feasibility
    # def fmkt4(self, Y_t, curlC_t, G_t, M_t):
    #     return Y_t - curlC_t - G_t - M_t

    ## --- Household block ---
    # This is the heterogenous part, so it's a little more complex. We use the
    # endeogenous grid method to solve for policy rules, follow Young to travel
    # between points
    def egm_back(self, V_a_p, W_t, R_t, D_t, eta_t, tauP_t, tauL_t):
        # egm backwards
        V_a_t, a_t, c_t, l_t = egm_back(
                V_a_p, W_t, R_t, D_t + eta_t, tauP_t, tauL_t,  # economy
                self.z_grid, self.beta, self.z_tran_mat, self.gamma, self.phi, self.chi, self.a_grid, self.a_min,  # params
            )
        
        # results
        return V_a_t, {'a': a_t, 'c': c_t, 'l': l_t}
    
    def agg_dist(self, x):
        return agg_dist(self.dist, x)

    def agg_l(self, l):
        return agg_dist(self.dist, l * self.z_grid)


    ## --- Config ---
    # These define how the steady state is found
    # config dictionaries
    # simple_blocks: dict points variable to evaluation function and inputs
    simple_blocks = {  # order matters, needs to be in an order that can be evaluated
            'I': (fI, inputI), 'R': (fR, inputR),  # monetary block
            'G': (fG, inputG), 'tauL': (ftauL, inputtauL),  # govt block
            'N': (fN, inputN), 'M': (fM, inputM), 'D': (fD, inputD),  # firm block
        }
    # ha_block: only one ha block, so in an order list. Has egm function, inputs, aggregators
    ha_block = {  # egm func
            'egm': egm_back,
            'inputs': ['W', 'R', 'D', 'eta', 'tauP', 'tauL'], 
            'aggs': {
                'curlB': (agg_dist, 'a'),
                'curlC': (agg_dist, 'c'),
                'curlN': (agg_l, 'l'),
            },
        }
    # markets: dict points market to evaluation function (goes to 0) and inputs
    markets = {
            'mkt1': (fmkt1, inputmkt1), 'mkt2': (fmkt2, inputmkt2),  # philips curve, bond law of motion
            'mkt3': (fmkt3, inputmkt3), 'mkt4': (fmkt4, inputmkt4),  # labor market, bonds market
        }
    vars = ['Y', 'pi', 'B', 'W']  # passed into the function, we need to know these for perturbation
    shocks = ['A', 'psi', 'g', 'xi', 'tauP', 'eta']
    exog = vars + shocks


# egm, do these outside the class to njit them
@nb.njit
def labor_leisure_choice(c_0, Wz_t_grid, T_t_grid,  # econ states
                         chi, phi, gamma,
                         maxiter=50, tol=1e-12):
    ## solves for optimal labor given consumption
    # setup loop
    c = c_0
    # newtons method
    f = lambda c: c - Wz_t_grid**(1 + 1 / chi) / phi**(1 / chi) * c**(-gamma / chi) - T_t_grid
    df = lambda c: 1 + (gamma / chi) * Wz_t_grid**(1 + 1 / chi) / phi**(1 / chi) * c**(-gamma / chi - 1)
    for _ in range(maxiter):
        # update rule (1d newton)
        dc = f(c) / df(c)
        c -= dc

        # exit condition
        if np.abs(dc).max() < tol:
            break
    else:
        raise RuntimeError(f'Labor-Leisure Choice: Iteration Count Exceeded. Curent Tolerance = {np.abs(dc).max()}')
    
    return c

@nb.njit
def egm_back(V_a_p, W_t, R_t, S_t, tauP_t, tauL_t,  # econ states
             z_grid, beta, z_tran_mat, gamma, phi, chi, a_grid, a_min,  # parameters
             ):
    ## value function derivates following Carroll based on eocnomic variables
    # at the gridpoints
    # setup
    T_t_grid = S_t - tauL_t * z_grid**tauP_t  # net transfers
    Wz_t_grid = W_t * z_grid  # wage at each gridpoints

    # get consumption
    dc_t_nextgrid = V_a_p @ (beta * z_tran_mat)
    c_t_nextgrid = (dc_t_nextgrid)**(-1 / gamma)  # get consumption (on next period grid)
    l_t_nextgrid = (Wz_t_grid * dc_t_nextgrid / phi)**(1 / chi)  # labor (on next period grid)
    Ra_l_nextgrid = a_grid[:, None] + c_t_nextgrid - Wz_t_grid * l_t_nextgrid - T_t_grid  # assets last period (on next period grid)

    # get values on current grid
    c_t = func_interp(R_t * a_grid, Ra_l_nextgrid, c_t_nextgrid)  # map to gridpoints now, by linearity can mutiply by R on the lhs instead of div on rhs
    l_t = func_interp(R_t * a_grid, Ra_l_nextgrid, l_t_nextgrid)

    # budget constraint
    a_t = R_t * a_grid[:, None] + Wz_t_grid * l_t + T_t_grid - c_t  # option 1
    constr_i, constr_j = np.where(a_t < a_min)  # mask for where households are borrowing constrained
    if len(constr_i) > 0:
        c_constr = np.empty(len(constr_i))
        for i in nb.prange(len(constr_i)):
            c_constr[i] = c_t[constr_i[i], constr_j[i]]
        T_constr = R_t * a_grid[constr_i] + T_t_grid[constr_j] - a_min  # transfers to constrained people
        c_constr = labor_leisure_choice(c_constr, Wz_t_grid[constr_j], T_constr, chi, phi, gamma)
        l_constr = (c_constr - T_constr) / Wz_t_grid[constr_j]
        
        # put into arrays
        for i in nb.prange(len(constr_i)):
            a_t[constr_i[i], constr_j[i]] = a_min
            c_t[constr_i[i], constr_j[i]] = c_constr[i]
            l_t[constr_i[i], constr_j[i]] = l_constr[i]

    # value function derivative (to go backwards again)
    V_a_t = R_t * c_t**(-gamma)

    return V_a_t, a_t, c_t, l_t


if __name__ == '__main__':
    # imports
    from toolkit.irfs import single_shock_irfs, decompose_single_shock_irfs, policy_irf, decompose_policy_irf
    import matplotlib.pyplot as plt


    ## --- Solve Steady State ---
    # hank parameters
    mod_pars = {
            'gamma': 2., 'chi': 2.,  # household parameters, beta and phi are chosen in calibration
            'N_z': 7, 'rho_z': 0.966, 'sigma_z': 0.129,  # household state transitions
            'kappa': 0.1,  # philips curve
            'rho_B': 0.95, 'omega_pi': 1.5, 'omega_Y': 0.,  # govt parameters
            'A': 1., 'psi': 1.2, 'g': 0.201, 'xi': 1., 'tauP': 2., 'eta': 0.081,  # shock steady states
            'pi': 1., 'B': 0.577,  # variable steady states we define (otherwise model can't be solved)
            'I': 1.005, 'Y': 1.,  # varaible steady states we target
            'N_a': 501, 'a_min': 0, 'a_max': 50,  # grid states
        }

    # hank object, everything is done by this!
    hank = HANK(**mod_pars)

    # calibration parameters
    free = ['beta', 'phi']
    endog = ['R', 'G', 'tauL', 'N', 'M', 'D', 'curlN', 'curlC', 'curlB']
    markets = {m: HANK.markets[m] for m in ['mkt3', 'mkt4']}  # the other markets clear by definition or by an analytic form
    V_a0 = (hank.a_grid[:, None] * (hank.I - 1) + hank.z_grid)**(-hank.gamma)
    X0 = np.array([0.99, 1.2])

    # calibrate the hank
    hank.solve_ss(free, endog, V_a0, X0, markets=markets)
    assert np.isclose(hank.curlC + hank.G - hank.Y, 0)  # walras

    ## policy function plot
    # setup plot
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    n = (hank.a_grid <= 1).sum()  # number of gridpoints we plot
    axs[0].set_title('Savings Rule')
    axs[0].set_xlabel('$a_{i, t-1}$')
    axs[0].set_ylabel('$a_{i, t}$')
    axs[1].set_title('Consumption Rule')
    axs[1].set_xlabel('$a_{i, t-1}$')
    axs[1].set_ylabel('$c_{i, t}$')
    axs[2].set_title('Labor Rule')
    axs[2].set_xlabel('$a_{i, t-1}$')
    axs[2].set_ylabel(r'$\ell_{i, t}$')

    # savings rule
    axs[0].plot(hank.a_grid[:n], hank.a[:n], label=['$z_{i, t}='+str(round(z, 2))+'$' for z in hank.z_grid])
    axs[0].plot([0, hank.a_grid[n - 1]], [0, hank.a_grid[n - 1]], 'k--', label='45 Degree Line')

    # consumoption rule
    axs[1].plot(hank.a_grid[:n], hank.c[:n])

    # labor rule
    axs[2].plot(hank.a_grid[:n], hank.l[:n])

    # final things
    fig.tight_layout()
    fig.subplots_adjust(right=12/14)
    fig.legend(loc='center left', bbox_to_anchor=(12/14, 0.5), frameon=False)

    
    ## --- Shock the Model ---
    # solve for the shock response matricies 
    T = 300  # time horizen for shocks
    G, dxs = hank.solve_G(T, ha=True)  # if you just want normal irfs, run with ha=False

    # find the irf
    irfs, exog_irf = single_shock_irfs(G['A'], 0.9, T)  # set rho = 0.9
    decomp = decompose_single_shock_irfs(G['A'], hank.int_block_jacs, exog_irf, 'A')  # cool plots

    # plot config
    fig, axs = plt.subplots(3, 4)
    fig.set_figwidth(16)
    fig.set_figheight(8)
    fig.suptitle('1% TFP Shock')
    fig.supxlabel('$t$')
    fig.supylabel('% Dev. From SS')

    # plot irfs
    vars = ['Y', 'pi', 'B', 'W', 'I', 'R', 'G', 'tauL', 'N', 'D', 'curlB', 'curlC']  # variables we're interested in
    labels = {
            'Y': '$Y$', 'pi': r'$\pi$', 'B': '$B$',
            'W': '$W$', 'I': '$I$', 'R': '$R$', 'M': '$M$',
            'G': '$G$', 'tauL': r'$\tau^L$', 'N': '$N$',
            'D': '$D$', 'curlB': r'$\mathcal{B}$', 'curlC': r'$\mathcal{C}$',
            'A': '$A$',
        }
    t = 100  # plot first 100 periods after shock
    for i in range(12):
        # variables for i
        v = vars[i]
        ax = axs.take(i)

        # plot irf
        ax.set_title(labels[v])
        ax.plot(irfs[v][:t] / hank.__getattribute__(v), label='Aggregate Effects')

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
    ts = [0, 10, 20]
    n = (hank.a_grid < 1).sum()  # number of gridpoints we plot

    # solve hh impulse response
    pol_irf = policy_irf(dxs['c'], G['A'], exog_irf, 'A')
    pol_decomp = decompose_policy_irf(dxs['c'], G['A'], exog_irf, 'A')

    # plot config
    fig, axs = plt.subplots(6, 3, sharey='row')
    fig.set_figwidth(14)
    fig.set_figheight(10)
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
        lns = axs[0, i].plot(hank.a_grid[:n], pol_irf[t, :n] / hank.c[:n], label=['$z='+str(round(z, 2))+'$' for z in hank.z_grid])

        # decomposition
        for j, (X, decomp) in enumerate(pol_decomp.items()):
            axs[1 + j, i].set_title(f'{labels[X]} Effects')
            axs[1 + j, i].plot(hank.a_grid[:n], pol_decomp[X][t, :n] / hank.c[:n])

    # final things
    fig.tight_layout()
    fig.subplots_adjust(right=8/10)
    fig.legend(loc='center left', bbox_to_anchor=(12/14, 0.5), frameon=False, handles=lns)


    # show the plot
    plt.show()
