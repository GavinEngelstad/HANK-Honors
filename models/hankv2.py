'''
A 1 asset, discrete time HANK model wtih 7 shocks applied. The model is outlined
in the paper. To solve it, I use the method from the Auclert et al. "Using the
Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models."

The model features a Heterogenous households and a block for firms, monetary policy,
the government, and unions to aggreagte labor.
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
         - Wage Philips Curve: kappa_W
         - Govt Parameters: rho_B, omega_pi, omega_Y
         - Shock Steady States: A, psi, psiW, g, xi, tauP, eta
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
    inputI = [('pi', 0), ('Y', 0), ('epsxi', 0)]
    def fI(self, pi_t, Y_t, epsxi_t):
        return self.I * (pi_t / self.pi)**self.omega_pi * (Y_t / self.Y)**self.omega_Y * epsxi_t * self.xi

    inputR = [('I', 1), ('pi', 0)]
    def fR(self, I_l, pi):
        return I_l / pi

    # govt block
    # Takes: g, Y, R, B, eta, tauP
    # Returns: G, tauL
    inputG = [('epsg', 0), ('Y', 0)]
    def fG(self, epsg_t, Y_t):
        return epsg_t * self.g * Y_t

    inputtauL = [('R', 0), ('B', 1), ('G', 0), ('B', 0), ('epseta', 0), ('epstauP', 0)]
    def ftauL(self, R_t, B_l, G_t, B_t, epseta_t, epstauP_t):
        T = agg_dist(self.z_dist, self.z_grid**(epstauP_t * self.tauP))  # tax progressivity
        return (R_t * B_l + G_t + epseta_t * self.eta - B_t) / T

    # firm block
    # Takes: Y, A, pi, W
    # Returns: N, M, D
    inputN = [('Y', 0), ('epsA', 0)]
    def fN(self, Y_t, epsA_t):
        return Y_t / self.A / epsA_t

    inputM = [('pi', 0), ('Y', 0)]
    def fM(self, pi_t, Y_t):  # M, FOC is 0 so we can prolly remove it?
        return self.kappa / 2 * (pi_t / self.pi - 1)**2 * Y_t

    inputD = [('Y', 0), ('W', 0), ('N', 0), ('M', 0)]
    def fD(self, Y_t, W_t, N_t, M_t):
        return Y_t - W_t * N_t - M_t
    
    # union block
    # Takes: W, N
    # Returns: piW, L
    inputpiW = [('W', 0), ('W', 1)]
    def fpiW(self, W_t, W_l):
        return W_t / W_l
    
    inputL = [('N', 0)]
    def fL(self, N):
        return N / self.z_agg

    ## --- Market Clearing ---
    # Essnetually just acts like a simple block
    # Takes: Y, W, pi, psi, A, R, G, eta, B, N, curlN, curlB  (curlX is X demand from HA block)
    # Returns: 0, 0, 0, 0 (hopefully) for philips curve, bond law of motion, labor market, and bond market
    inputmkt1 = [('pi', 0), ('pi', -1), ('epspsi', 0), ('W', 0), ('epsA', 0), ('R', -1), ('Y', -1), ('Y', 0)]
    def fmkt1(self, pi_t, pi_p, epspsi_t, W_t, epsA_t, R_p, Y_p, Y_t):
        hatpi_t = pi_t / self.pi - 1  # log deviation from steady state
        hatpi_p = pi_p / self.pi - 1
        # lhs = self.kappa * hatpi_t * (hatpi_t - 1)
        # rhs = 1 - psi_t + psi_t * W_t / A_t + self.kappa / R_p * Y_p / Y_t * hatpi_p * (hatpi_p - 1)
        return Y_p / Y_t / R_p * hatpi_p + self.kappa * (W_t / epsA_t / self.A - 1 / epspsi_t / self.psi) - hatpi_t

    inputmkt2 = [('B', 0), ('R', 0), ('B', 1), ('G', 0), ('epseta', 0)]
    def fmkt2(self, B_t, R_t, B_l, G_t, epseta_t):
        RBdiff = R_t * B_l - self.R * self.B  # level devation from steady state
        Gdiff = G_t - self.G
        etadiff = epseta_t * self.eta - self.eta
        return B_t - self.B - self.rho_B * (RBdiff + Gdiff + etadiff)

    inputmkt3 = [('piW', 0), ('piW', -1), ('L', 0), ('W', 0), ('curlCZ', 0), ('epspsiW', 0)]
    def fmkt3(self, piW_t, piW_p, L_t, W_t, curlCZ_t, epspsiW_t):
        hatpiW_t = piW_t / self.piW - 1
        hatpiW_p = piW_p / self.piW - 1
        return self.kappa_W * (self.phi * L_t**(1 + self.chi) -  W_t * L_t * curlCZ_t / epspsiW_t / self.psiW) + self.beta * hatpiW_p - hatpiW_t

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
    def egm_back(self, V_a_p, L_t, W_t, R_t, D_t, epseta_t, epstauP_t, tauL_t):
        # egm backwards
        V_a_t, a_t, c_t, cz_t = egm_back(
                V_a_p, L_t, W_t, R_t, D_t + epseta_t * self.eta, epstauP_t * self.tauP, tauL_t,  # economy
                self.z_grid, self.beta, self.z_tran_mat, self.gamma, self.a_grid, self.N_z,  # params
            )
        
        # results
        return V_a_t, {'a': a_t, 'c': c_t, 'cz': cz_t}
    
    def agg_dist(self, x):
        return agg_dist(self.dist, x)


    ## --- Config ---
    # These define how the steady state is found
    # config dictionaries
    # simple_blocks: dict points variable to evaluation function and inputs
    simple_blocks = {  # order matters, needs to be in an order that can be evaluated
            'I': (fI, inputI), 'R': (fR, inputR),  # monetary block
            'G': (fG, inputG), 'tauL': (ftauL, inputtauL),  # govt block
            'N': (fN, inputN), 'M': (fM, inputM), 'D': (fD, inputD),  # firm block
            'piW': (fpiW, inputpiW),  'L': (fL, inputL),  # union block
        }
    # ha_block: only one ha block, so in an order list. Has egm function, inputs, aggregators
    ha_block = {  # egm func
            'egm': egm_back,
            'inputs': ['L', 'W', 'R', 'D', 'epseta', 'epstauP', 'tauL'], 
            'aggs': {
                'curlB': (agg_dist, 'a'),
                'curlC': (agg_dist, 'c'),
                'curlCZ': (agg_dist, 'cz'),
            },
        }
    # markets: dict points market to evaluation function (goes to 0) and inputs
    markets = {
            'mkt1': (fmkt1, inputmkt1), 'mkt2': (fmkt2, inputmkt2),  # philips curve, bond law of motion
            'mkt3': (fmkt3, inputmkt3), 'mkt4': (fmkt4, inputmkt4),  # labor market, bonds market
        }
    vars = ['Y', 'pi', 'B', 'W']  # passed into the function, we need to know these for perturbation
    shocks = ['epsA', 'epspsi', 'epspsiW', 'epsg', 'epsxi', 'epstauP', 'epseta']
    exog = vars + shocks


# egm, do these outside the class to njit them
@nb.njit
def egm_back(V_a_p, L_t, W_t, R_t, S_t, tauP_t, tauL_t,  # econ states
             z_grid, beta, z_tran_mat, gamma, a_grid, N_z,  # parameters
             ):
    ## value function derivates following Carroll based on eocnomic variables
    # at the gridpoints
    # setup
    T_t_grid = S_t - tauL_t * z_grid**tauP_t  # net transfers
    Wz_t_grid = W_t * z_grid * L_t  # earnings at each gridpoint

    # get consumption
    dc_t_nextgrid = V_a_p @ (beta * z_tran_mat)
    c_t_nextgrid = (dc_t_nextgrid)**(-1 / gamma)  # get consumption (on next period grid)
    Ra_l_nextgrid = a_grid[:, None] + c_t_nextgrid - Wz_t_grid - T_t_grid  # assets last period (on next period grid)

    # get values on current grid
    a_t = func_interp(R_t * a_grid, Ra_l_nextgrid, a_grid.repeat(N_z).reshape((-1, N_z)))  # map to gridpoints now, by linearity can mutiply by R on the lhs instead of div on rhs
    c_t = Wz_t_grid + R_t * a_grid[:, None] + T_t_grid - a_t
    dc_t = c_t**(-gamma)

    # value function derivative (to go backwards again)
    V_a_t = R_t * dc_t

    return V_a_t, a_t, c_t, dc_t * z_grid


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
