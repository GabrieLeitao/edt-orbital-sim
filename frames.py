import numpy as np
from numba import njit

@njit(fastmath=True)
def get_rotation_matrix_eci(inc):
    """
    Returns the rotation matrix from Orbital Plane to ECI frame.
    Assumes RAAN=0 and Arg_Per=0 for simplicity (KISS).
    The orbit starts at the Ascending Node on the X-axis.
    
    For 0 <= inc < 90, this produces a prograde orbit moving Northward (v_z > 0) at t=0.
    Matrix: Rx(inc)
    """
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_i, -sin_i],
        [0.0, sin_i, cos_i]
    ])

@njit(fastmath=True)
def get_earth_rotation_components(t):
    """
    Calculates Earth rotation components for ECI/ECEF transformations.
    Returns (cos_tg, sin_tg)
    """
    omega_e = 7.2921151467e-5
    theta_g0 = 0.0 
    theta_gmst = (theta_g0 + omega_e * t) % (2 * np.pi)
    return np.cos(theta_gmst), np.sin(theta_gmst)

@njit(fastmath=True)
def eci_to_ecef(r_eci, cos_tg, sin_tg):
    """
    Transforms a vector from ECI to ECEF frame.
    """
    return np.array([
        cos_tg * r_eci[0] + sin_tg * r_eci[1],
        -sin_tg * r_eci[0] + cos_tg * r_eci[1],
        r_eci[2]
    ])

@njit(fastmath=True)
def ecef_to_eci(r_ecef, cos_tg, sin_tg):
    """
    Transforms a vector from ECEF to ECI frame.
    """
    return np.array([
        cos_tg * r_ecef[0] - sin_tg * r_ecef[1],
        sin_tg * r_ecef[0] + cos_tg * r_ecef[1],
        r_ecef[2]
    ])

@njit(fastmath=True)
def eci_to_lvlh(r_eci, v_eci, r_target_eci):
    """
    Transforms position vectors from ECI to LVLH frame centered on a target.
    Optimized for Numba with explicit loops and avoided vstack.
    """
    # 1. Define LVLH unit vectors based on target state
    r_mag = np.linalg.norm(r_target_eci)
    u_z = -r_target_eci / r_mag
    
    h = np.cross(r_target_eci, v_eci)
    h_norm = np.linalg.norm(h)
    
    if h_norm < 1e-6:
        # Fallback for degenerate orbits
        if np.abs(u_z[2]) > 0.9:
            u_y = np.array([1.0, 0.0, 0.0])
        else:
            u_y = np.array([0.0, 0.0, 1.0])
        u_x = np.cross(u_y, u_z)
        u_y = np.cross(u_z, u_x) # Re-orthogonalize
    else:
        u_y = -h / h_norm
        u_x = np.cross(u_y, u_z)
    
    # 2. Manual rotation to avoid np.dot on large matrices if needed
    # but for small arrays it's fine.
    num_points = r_eci.shape[0]
    r_lvlh = np.zeros((num_points, 3))
    
    for i in range(num_points):
        rx = r_eci[i, 0] - r_target_eci[0]
        ry = r_eci[i, 1] - r_target_eci[1]
        rz = r_eci[i, 2] - r_target_eci[2]
        
        r_lvlh[i, 0] = rx * u_x[0] + ry * u_x[1] + rz * u_x[2]
        r_lvlh[i, 1] = rx * u_y[0] + ry * u_y[1] + rz * u_y[2]
        r_lvlh[i, 2] = rx * u_z[0] + ry * u_z[1] + rz * u_z[2]
        
    return r_lvlh
