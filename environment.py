import numpy as np
from numba import njit
import params as p

@njit
def get_environment_fast(r, v, t, p_arr):
    """
    Numba-optimized environment function.
    Returns (B_vec, rho, I) as a tuple for speed.
    """
    r_norm = np.linalg.norm(r)
    u_r = r / r_norm
    
    # 1. Magnetic Field (B) - Simple Tilted Dipole
    B_0 = 3.12e-5  # Tesla at equator
    # B = B_0 * (RE / r_norm)^3 * (3 * u_r_z * u_r - [0, 0, 1])
    B = B_0 * (p_arr[p.IDX_RE] / r_norm)**3 * (3 * u_r[2] * u_r - np.array([0.0, 0.0, 1.0]))
    
    # 2. Atmospheric Density (rho) - Exponential model
    h = r_norm - p_arr[p.IDX_RE]
    rho_0 = 1.225  # Sea level [kg/m^3]
    H = 8500.0     # Scale height [m]
    rho = rho_0 * np.exp(-h / H)
    
    # 3. Current (I)
    I = p_arr[p.IDX_I_EDT]
    
    return B, rho, I
