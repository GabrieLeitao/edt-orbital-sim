import numpy as np
from numba import njit
import params as p

@njit
def get_environment_fast(r, v, t, p_arr):
    """
    Numba-optimized environment function.
    Returns (B_vec, rho) as a tuple for speed.
    """
    r_norm = np.linalg.norm(r)
    u_r = r / r_norm
    
    # 1. Magnetic Field (B) - Simple Centered Dipole (Z-aligned)
    # B = B_0 * (RE / r_norm)^3 * ([0, 0, 1] - 3 * u_r_z * u_r)
    # Note: This is an engineering approximation. For scientific-grade 
    # orbital simulation, use IGRF (International Geomagnetic Reference Field).
    B_0 = 3.12e-5  # Tesla at equator
    B = B_0 * (p_arr[p.IDX_RE] / r_norm)**3 * (np.array([0.0, 0.0, 1.0]) - 3 * u_r[2] * u_r)
    
    # 2. Atmospheric Density (rho) - LEO Model
    # Exponential model using 500km as reference for better LEO fidelity
    # Rho at 500km is approx 5e-13 kg/m^3
    h = r_norm - p_arr[p.IDX_RE]
    rho_ref = 5.0e-13 # Reference density at 500km [kg/m^3]
    h_ref = 500000.0   # Reference altitude [m]
    H_leo = 65000.0    # Scale height in LEO [m] (approx 65km)
    
    rho = rho_ref * np.exp(-(h - h_ref) / H_leo)
    
    return B, rho
