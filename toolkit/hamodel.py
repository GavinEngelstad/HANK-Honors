'''
Parent class for HA models. Allows us to find steady states and apply shocks
to the model.
'''

## load some packages
from toolkit.grids import asset_grid, idiosyncratic_grid


## class
class HAModel:
    '''
    Model class

    Parent class for HA models. Handles the gridpoints, steady state finding
    (through another class) and perturbations (again, through another class).

    Models should initialize this with an `N_a` and `N_z` attribute for the number
    of asset gridpoints and idiosyncratic gridpioints respectively. The asset grid
    should have a defined `a_min` and `a_max` to interpolate between and the
    idosyncratic grid should have an rho_z and sigma_z for the persistance and 
    size of idosyncratic movements in logs.
    '''

    def __init__(self, N_a, a_min, a_max, N_z, rho_z, sigma_z):
        '''
        Defines the gridpoints the model is solved on
        '''
        # store values
        self.N_a = N_a  # asset grid things
        self.a_min = a_min
        self.a_max = a_max
        self.N_z = N_z  # idiosyncratic grid things
        self.rho_z = rho_z
        self.sigma_z = sigma_z

        # asset grid
        self.a_grid = asset_grid(N_a, a_min, a_max)

        # idiosyncratic grid
        self.z_grid, self.z_dist, self.z_tran_mat, self.z_agg = idiosyncratic_grid(N_z, rho_z, sigma_z)
