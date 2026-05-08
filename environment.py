import numpy as np
from numba import njit
import params as p

@njit(fastmath=True)
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
    
    # --- D. ATMOSPHERIC DENSITY (rho) ---
    # Scientific Model: Multi-layer Exponential + Diurnal Bulge
    # This accounts for both altitude-dependent scale height and day/night variations.
    h_km = (r_norm - p_arr[p.IDX_RE]) / 1000.0
    rho_base = get_density_standard(h_km)
    
    sun_dir = get_sun_direction(t)
    rho = apply_diurnal_bulge(rho_base, r_eci, sun_dir)
    
    return B_eci, rho

@njit(fastmath=True)
def get_sun_direction(t):
    """
    Approximate Sun direction in ECI frame.
    Assumes t=0 is Vernal Equinox for simplicity.
    """
    # Earth orbital period in seconds
    year_s = 365.25 * 86400.0
    lambda_sun = (2.0 * np.pi * t / year_s)
    epsilon = 0.40909 # Obliquity of ecliptic (23.44 degrees) in radians
    
    s = np.array([
        np.cos(lambda_sun),
        np.sin(lambda_sun) * np.cos(epsilon),
        np.sin(lambda_sun) * np.sin(epsilon)
    ])
    return s / np.linalg.norm(s)

@njit(fastmath=True)
def apply_diurnal_bulge(rho_base, r_eci, sun_dir):
    """
    Applies a Harris-Priester inspired diurnal bulge correction.
    Accounts for the density increase on the day-side, lagged by ~2 hours.
    """
    # Bulge lag (approx 30 degrees / 2 hours behind the sun)
    lag = -0.5236 # -30 degrees in radians
    cos_l, sin_l = np.cos(lag), np.sin(lag)
    
    # Rotate sun direction to find bulge center (Z-axis rotation)
    bulge_dir = np.array([
        cos_l * sun_dir[0] - sin_l * sun_dir[1],
        sin_l * sun_dir[0] + cos_l * sun_dir[1],
        sun_dir[2]
    ])
    
    r_unit = r_eci / np.linalg.norm(r_eci)
    cos_phi = np.dot(r_unit, bulge_dir)
    
    # Diurnal factor: increases with altitude
    # At 200km, variation is small; at 800km, it can be 5x-10x.
    h_km = (np.linalg.norm(r_eci) - 6378137.0) / 1000.0
    # Empirical scaling for the bulge amplitude
    delta = 0.002 * (h_km - 150.0) 
    delta = max(0.0, delta) # No bulge below 150km
    
    # Harris-Priester interpolation factor (cos^n(phi/2))
    # We use n=3 for a realistic bulge shape
    phi_factor = ((1.0 + cos_phi) / 2.0)**3
    
    return rho_base * (1.0 + delta * phi_factor)

@njit(fastmath=True)
def get_density_standard(h_km):
    """
    Multi-layer exponential atmospheric density model.
    Data sourced from Vallado, Table 8-4 (Standard Atmosphere).
    Covers 0 to 1000 km.
    """
    # Reference Altitudes [km]
    h_ref = np.array([
        0, 25, 50, 75, 100, 110, 120, 130, 140, 150, 
        180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000
    ])
    
    # Reference Densities [kg/m^3]
    rho_ref = np.array([
        1.225, 3.899e-2, 1.027e-3, 8.145e-5, 5.604e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 2.076e-9,
        5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13, 
        3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15
    ])
    
    # Scale Heights [km]
    H_ref = np.array([
        7.249, 6.349, 6.682, 7.110, 5.852, 7.263, 9.473, 12.636, 16.149, 22.523,
        29.740, 37.105, 45.546, 53.628, 62.151, 71.835, 82.463, 94.651, 124.64, 155.81, 195.42, 263.21, 361.02
    ])
    
    # Find the appropriate layer
    if h_km < 0:
        return 1.225 # Sea level fallback
    if h_km >= 1000:
        # Extrapolate using the last scale height
        return rho_ref[-1] * np.exp(-(h_km - h_ref[-1]) / H_ref[-1])
        
    # Standard lookup
    idx = 0
    for i in range(len(h_ref) - 1):
        if h_km >= h_ref[i] and h_km < h_ref[i+1]:
            idx = i
            break
            
    # Exponential interpolation: rho = rho0 * exp(-(h - h0) / H)
    return rho_ref[idx] * np.exp(-(h_km - h_ref[idx]) / H_ref[idx])

# ---------------------------------------------------------
# 1. HARDCODED IGRF-2000 COEFFICIENTS (Degree 4)
# Captures ~99% of the field's influence on LEO spacecraft.
# Values are in nanoTeslas (nT).
# ---------------------------------------------------------
@njit(fastmath=True)
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
@njit(fastmath=True)
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