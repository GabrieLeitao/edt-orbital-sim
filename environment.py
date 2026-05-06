import numpy as np
from numba import njit
import params as p

@njit
def get_environment_fast(r_eci, v, t, p_arr):
    """
    Numba-optimized environment function.
    Returns (B_vec, rho) as a tuple for speed.
    """
    r_norm = np.linalg.norm(r_eci)
    
    # --- A. FRAME TRANSFORMATION (ECI to ECEF) ---
    omega_e = 7.2921151467e-5  # Earth's rotation rate [rad/s]
    
    # Arbitrary starting angle for the Earth at t=0
    # You can change this if you want to test starting over different longitudes
    theta_g0 = 0.0 
    
    # Earth rotation angle based purely on your simulation time
    theta_gmst = (theta_g0 + omega_e * t) % (2 * np.pi)
    
    cos_tg = np.cos(theta_gmst)
    sin_tg = np.sin(theta_gmst)
    
    # Rotate r_eci to r_ecef (Z-axis rotation)
    r_ecef = np.array([
        cos_tg * r_eci[0] + sin_tg * r_eci[1],
        -sin_tg * r_eci[0] + cos_tg * r_eci[1],
        r_eci[2]
    ])
    
    # --- B. COMPUTE MAGNETIC FIELD ---
    g, h = get_igrf2000_coeffs()
    B_ecef = compute_igrf_ecef(r_ecef, g, h)
    
    # --- C. FRAME TRANSFORMATION (ECEF back to ECI) ---
    # Rotate B_ecef back to inertial frame so EDT cross products work
    B_eci = np.array([
        cos_tg * B_ecef[0] - sin_tg * B_ecef[1],
        sin_tg * B_ecef[0] + cos_tg * B_ecef[1],
        B_ecef[2]
    ])
    
    # 2. Atmospheric Density (rho) - LEO Model
    # Exponential model using 500km as reference for better LEO fidelity
    # Rho at 500km is approx 5e-13 kg/m^3
    h = r_norm - p_arr[p.IDX_RE]
    rho_ref = 5.0e-13 # Reference density at 500km [kg/m^3]
    h_ref = 500000.0   # Reference altitude [m]
    H_leo = 65000.0    # Scale height in LEO [m] (approx 65km)
    
    rho = rho_ref * np.exp(-(h - h_ref) / H_leo)
    
    return B_eci, rho

# ---------------------------------------------------------
# 1. HARDCODED IGRF-2000 COEFFICIENTS (Degree 4)
# Captures ~99% of the field's influence on LEO spacecraft.
# Values are in nanoTeslas (nT).
# ---------------------------------------------------------
@njit
def get_igrf2000_coeffs():
    # Arrays are size (5, 5) to accommodate n=0..4, m=0..4
    g = np.zeros((5, 5))
    h = np.zeros((5, 5))
    
    # Degree 1 (Dipole)
    g[1,0] = -29615.0; g[1,1] = -1728.0; h[1,1] = 5186.0
    # Degree 2 (Quadrupole)
    g[2,0] = -2267.0;  g[2,1] = 3072.0;  h[2,1] = -2246.0; g[2,2] = 1672.0; h[2,2] = -286.0
    # Degree 3 (Octupole)
    g[3,0] = 1341.0;   g[3,1] = -2290.0; h[3,1] = -227.0;  g[3,2] = 1253.0; h[3,2] = 296.0; g[3,3] = 715.0; h[3,3] = -492.0
    # Degree 4 (Hexadecapole - models the SAA depth)
    g[4,0] = 935.0;    g[4,1] = 787.0;   h[4,1] = 272.0;   g[4,2] = 251.0;  h[4,2] = -232.0; g[4,3] = -405.0; h[4,3] = 119.0; g[4,4] = 110.0; h[4,4] = -304.0
    
    return g, h

# ---------------------------------------------------------
# 2. IGRF MAGNETIC FIELD CALCULATOR (ECEF)
# ---------------------------------------------------------
@njit
def compute_igrf_ecef(r_ecef, g, h):
    RE = 6371200.0  # Earth reference radius in meters
    r_mag = np.linalg.norm(r_ecef)
    
    # Spherical coordinates
    theta = np.arccos(r_ecef[2] / r_mag)  # Co-latitude (0 to pi)
    phi = np.arctan2(r_ecef[1], r_ecef[0]) # Longitude (-pi to pi)
    
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    # B-field components in spherical coordinates
    Br, Bt, Bp = 0.0, 0.0, 0.0
    
    # Precompute sin/cos for longitude (m * phi)
    cos_m_phi = np.zeros(5)
    sin_m_phi = np.zeros(5)
    for m in range(5):
        cos_m_phi[m] = np.cos(m * phi)
        sin_m_phi[m] = np.sin(m * phi)

    # Schmidt Semi-Normalized Legendre Polynomials (P) and derivatives (dP)
    P = np.zeros((5, 5))
    dP = np.zeros((5, 5))
    P[0, 0] = 1.0
    
    # Compute P and dP up to degree 4
    for n in range(1, 5):
        for m in range(n + 1):
            if n == m:
                P[n, n] = sin_th * P[n-1, n-1]
                dP[n, n] = sin_th * dP[n-1, n-1] + cos_th * P[n-1, n-1]
                if m == 1:
                    P[n, n] *= np.sqrt(2.0)
                    dP[n, n] *= np.sqrt(2.0)
            elif n == 1 and m == 0:
                P[1, 0] = cos_th * P[0, 0]
                dP[1, 0] = cos_th * dP[0, 0] - sin_th * P[0, 0]
            else:
                K = ((n - 1)**2 - m**2) / (n**2 - m**2)
                factor1 = (2 * n - 1) / np.sqrt(n**2 - m**2)
                factor2 = np.sqrt(K)
                
                P[n, m] = factor1 * cos_th * P[n-1, m] - factor2 * P[n-2, m]
                dP[n, m] = factor1 * (cos_th * dP[n-1, m] - sin_th * P[n-1, m]) - factor2 * dP[n-2, m]

    # Summing the Spherical Harmonics
    rho = RE / r_mag
    for n in range(1, 5):
        pow_rho = rho**(n + 2)
        for m in range(n + 1):
            coef_cos = g[n, m] * cos_m_phi[m] + h[n, m] * sin_m_phi[m]
            coef_sin = g[n, m] * sin_m_phi[m] - h[n, m] * cos_m_phi[m]
            
            # Radial component
            Br += pow_rho * (n + 1) * coef_cos * P[n, m]
            # Theta (co-latitude) component
            Bt -= pow_rho * coef_cos * dP[n, m]
            # Phi (longitude) component
            if sin_th > 1e-10: # Avoid singularity at poles
                Bp -= pow_rho * m * coef_sin * P[n, m] / sin_th

    # Convert B from nT to Tesla
    Br *= 1e-9
    Bt *= 1e-9
    Bp *= 1e-9

    # Convert Spherical B-field to ECEF Cartesian
    Bx_ecef = Br * sin_th * cos_m_phi[1] + Bt * cos_th * cos_m_phi[1] - Bp * sin_m_phi[1]
    By_ecef = Br * sin_th * sin_m_phi[1] + Bt * cos_th * sin_m_phi[1] + Bp * cos_m_phi[1]
    Bz_ecef = Br * cos_th - Bt * sin_th

    return np.array([Bx_ecef, By_ecef, Bz_ecef])