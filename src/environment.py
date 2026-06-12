import numpy as np
from numba import njit
import params as p
from frames import get_earth_rotation_components, eci_to_ecef, ecef_to_eci


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
def harris_priester_density(alt_km, cos_phi):
    """
    Harris-Priester atmospheric density model with diurnal variation.
    Based on Vallado / Long et al. 1989 table.
    
    Parameters:
    -----------
    alt_km : float
        Altitude in km (100 to 1000 km, clamped to range)
    cos_phi : float
        Cosine of angle between position vector and bulge center
        (bulge center lags Sun by ~30 degrees)
    
    Returns:
    --------
    rho : float
        Atmospheric density in kg/m^3
    """
    
    # Harris-Priester table from Vallado
    altitudes = np.array([
        100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 280.0, 290.0, 300.0, 320.0,
        340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0,
        540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0, 720.0,
        760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0, 1000.0
    ])
    
    min_dens = np.array([
        4.974e-07, 2.490e-08, 8.377e-09, 3.899e-09, 2.122e-09, 1.263e-09,
        8.008e-10, 5.283e-10, 3.617e-10, 2.557e-10, 1.839e-10, 1.341e-10,
        9.949e-11, 7.488e-11, 5.709e-11, 4.403e-11, 2.697e-11, 2.139e-11,
        1.708e-11, 1.099e-11, 7.214e-12, 4.824e-12, 3.274e-12, 2.249e-12,
        1.558e-12, 1.091e-12, 7.701e-13, 5.474e-13, 3.916e-13, 2.819e-13,
        2.042e-13, 1.488e-13, 1.092e-13, 8.070e-14, 6.012e-14, 4.519e-14,
        3.430e-14, 2.620e-14, 2.043e-14, 1.607e-14, 1.036e-14, 8.496e-15,
        7.069e-15, 4.680e-15, 3.200e-15, 2.210e-15, 1.560e-15, 1.150e-15
    ])
    
    max_dens = np.array([
        4.974e-07, 2.490e-08, 8.710e-09, 4.059e-09, 2.215e-09, 1.344e-09,
        8.758e-10, 6.010e-10, 4.297e-10, 3.162e-10, 2.396e-10, 1.853e-10,
        1.455e-10, 1.157e-10, 9.308e-11, 7.555e-11, 5.095e-11, 4.226e-11,
        3.526e-11, 2.511e-11, 1.819e-11, 1.337e-11, 9.955e-12, 7.492e-12,
        5.684e-12, 4.355e-12, 3.362e-12, 2.612e-12, 2.042e-12, 1.605e-12,
        1.267e-12, 1.005e-12, 7.997e-13, 6.390e-13, 5.123e-13, 4.121e-13,
        3.325e-13, 2.691e-13, 2.185e-13, 1.779e-13, 1.190e-13, 9.776e-14,
        8.059e-14, 5.741e-14, 4.210e-14, 3.130e-14, 2.360e-14, 1.810e-14
    ])
    
    # Clamp altitude to table range
    if alt_km < 100.0:
        alt_km = 100.0
    elif alt_km > 1000.0:
        alt_km = 1000.0
    
    # Find interpolation index (linear search - fast for small table)
    idx = 0
    for i in range(len(altitudes) - 1):
        if alt_km <= altitudes[i + 1]:
            idx = i
            break
    else:
        idx = len(altitudes) - 2
    
    # Linear interpolation for min and max densities
    t = (alt_km - altitudes[idx]) / (altitudes[idx + 1] - altitudes[idx])
    
    rho_min = min_dens[idx] + t * (min_dens[idx + 1] - min_dens[idx])
    rho_max = max_dens[idx] + t * (max_dens[idx + 1] - max_dens[idx])
    
    # Harris-Priester diurnal factor (n=2)
    # cos_phi = 1 at bulge center, -1 at opposite side
    phi_factor = ((1.0 + cos_phi) / 2.0) ** 2
    
    return rho_min + (rho_max - rho_min) * phi_factor


@njit(fastmath=True)
def get_density_standard(h_km):
    """
    Multi-layer exponential atmospheric density model.
    Data sourced from Vallado 2013, Table 8-4.
    - 0 km: US Standard Atmosphere 1976.
    - 25 to 500 km: CIRA-72.
    - 500 to 1000 km: CIRA-72 with exospheric temperature T_inf = 1000 K.
    """
    # Reference Altitudes [km]
    h_ref = np.array([
        0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 
        180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000
    ], dtype=np.float64)
    
    # Reference Densities [kg/m^3]
    rho_ref = np.array([
        1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 
        9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 
        9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15
    ], dtype=np.float64)
    
    # Scale Heights [km]
    H_ref = np.array([
        7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 
        7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 
        58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00
    ], dtype=np.float64)
    
    if h_km < 0:
        return 1.225
    if h_km >= 1000:
        return rho_ref[-1] * np.exp(-(h_km - h_ref[-1]) / H_ref[-1])
        
    # Optimized binary search
    idx = np.searchsorted(h_ref, h_km, side='right') - 1
    
    return rho_ref[idx] * np.exp(-(h_km - h_ref[idx]) / H_ref[idx])


@njit(fastmath=True)
def get_environment_optimized(r_eci, v, t, p_arr, cos_tg, sin_tg):
    """
    Optimized environment function using pre-calculated rotation components.
    Uses Harris-Priester density model for high fidelity.
    """
    r_norm = np.linalg.norm(r_eci)
    
    # --- A. FRAME TRANSFORMATION (ECI to ECEF) ---
    r_ecef = eci_to_ecef(r_eci, cos_tg, sin_tg)
    
    # --- B. COMPUTE MAGNETIC FIELD ---
    B_ecef = compute_igrf_ecef_fast(r_ecef)
    
    # --- C. FRAME TRANSFORMATION (ECEF back to ECI) ---
    B_eci = ecef_to_eci(B_ecef, cos_tg, sin_tg)
    
    # --- D. ATMOSPHERIC DENSITY (rho) ---
    alt_km = (r_norm - p_arr[p.IDX_RE]) / 1000.0

    if alt_km < 150.0:
        return B_eci, get_density_standard(alt_km)
    
    # Get Sun direction
    sun_dir = get_sun_direction(t)
    
    # Harris-Priester Bulge center lag (~30 degrees / 2 hours)
    # lag = -0.5236; cos and sin of -30 degrees
    cos_l, sin_l = 0.8660254037844387, -0.49999999999999994
    
    # Compute cos(phi) = angle between position and bulge center
    bulge_dir_x = cos_l * sun_dir[0] - sin_l * sun_dir[1]
    bulge_dir_y = sin_l * sun_dir[0] + cos_l * sun_dir[1]
    bulge_dir_z = sun_dir[2]
    
    cos_phi = (r_eci[0]*bulge_dir_x + r_eci[1]*bulge_dir_y + r_eci[2]*bulge_dir_z) / r_norm
    
    # Clamp for numerical stability
    if cos_phi > 1.0:
        cos_phi = 1.0
    elif cos_phi < -1.0:
        cos_phi = -1.0
        
    rho = harris_priester_density(alt_km, cos_phi)
    
    return B_eci, rho


@njit(fastmath=True)
def compute_igrf_ecef_fast(r_ecef):
    """
    Highly optimized IGRF calculator.
    Uses hardcoded coefficients to avoid array creation and binary search.
    """
    RE = 6371200.0
    r_mag = np.linalg.norm(r_ecef)
    
    # Spherical coordinates
    theta = np.arccos(r_ecef[2] / r_mag)
    phi = np.arctan2(r_ecef[1], r_ecef[0])
    
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    # Schmidt Semi-Normalized Legendre Polynomials (Pre-allocated or hardcoded for degree 4)
    # We use a flat array and manual indexing for speed in Numba
    P = np.zeros(15) # For degree 4: (n+1)(n+2)/2 = 15 elements
    dP = np.zeros(15)
    
    P[0] = 1.0 # P(0,0)
    
    # Hardcoded IGRF-2000 degree 1 coefficients
    g10 = -29615.0; g11 = -1728.0; h11 = 5186.0
    # Degree 2
    g20 = -2267.0;  g21 = 3072.0;  h21 = -2246.0; g22 = 1672.0; h22 = -286.0
    # Degree 3
    g30 = 1341.0;   g31 = -2290.0; h31 = -227.0;  g32 = 1253.0; h32 = 296.0; g33 = 715.0; h33 = -492.0
    # Degree 4
    g40 = 935.0;    g41 = 787.0;   h41 = 272.0;   g42 = 251.0;  h42 = -232.0; g43 = -405.0; h43 = 119.0; g44 = 110.0; h44 = -304.0

    # Legendre recurrence (simplified for degree 4)
    # n=1
    P[1] = cos_th  # P(1,0)
    dP[1] = -sin_th
    P[2] = np.sqrt(2.0) * sin_th # P(1,1)
    dP[2] = np.sqrt(2.0) * cos_th
    
    # n=2
    # P(2,0) = 0.5*(3*cos^2-1)
    P[3] = 1.5 * cos_th**2 - 0.5
    dP[3] = -3.0 * cos_th * sin_th
    # P(2,1) = sqrt(3)*sin*cos
    P[4] = np.sqrt(3.0) * sin_th * cos_th
    dP[4] = np.sqrt(3.0) * (cos_th**2 - sin_th**2)
    # P(2,2) = 0.5*sqrt(3)*sin^2
    P[5] = 0.5 * np.sqrt(3.0) * sin_th**2
    dP[5] = np.sqrt(3.0) * sin_th * cos_th
    
    # We stop here for brevity in the replacement, but the loop below is also njitted and fast.
    # The key is avoiding the constant array creation for g, h.
    
    # Use a loop for n=3,4 to keep code manageable while staying fast
    for n in range(3, 5):
        for m in range(n + 1):
            idx = n*(n+1)//2 + m
            idx_n1_m = (n-1)*n//2 + m
            idx_n2_m = (n-2)*(n-1)//2 + m
            if n == m:
                P[idx] = np.sqrt((2*n-1.0)/(2*n)) * sin_th * P[idx_n1_m]
                dP[idx] = np.sqrt((2*n-1.0)/(2*n)) * (sin_th * dP[idx_n1_m] + cos_th * P[idx_n1_m])
            else:
                K = ((n - 1.0)**2 - m**2) / (n**2 - m**2)
                f1 = (2*n - 1.0) / np.sqrt(n**2 - m**2)
                f2 = np.sqrt(K)
                P[idx] = f1 * cos_th * P[idx_n1_m] - f2 * P[idx_n2_m]
                dP[idx] = f1 * (cos_th * dP[idx_n1_m] - sin_th * P[idx_n1_m]) - f2 * dP[idx_n2_m]

    Br, Bt, Bp = 0.0, 0.0, 0.0
    rho = RE / r_mag
    
    # Manual unrolling of the sum for n=1..4
    # (n=1)
    pow_rho = rho**3
    # m=0
    c_phi = 1.0; s_phi = 0.0
    coef_cos = g10 * c_phi
    Br += pow_rho * 2 * coef_cos * P[1]
    Bt -= pow_rho * coef_cos * dP[1]
    # m=1
    c_phi = np.cos(phi); s_phi = np.sin(phi)
    coef_cos = g11 * c_phi + h11 * s_phi
    coef_sin = g11 * s_phi - h11 * c_phi
    Br += pow_rho * 2 * coef_cos * P[2]
    Bt -= pow_rho * coef_cos * dP[2]
    if sin_th > 1e-10: Bp -= pow_rho * 1 * coef_sin * P[2] / sin_th
    
    # (n=2)
    pow_rho = rho**4
    # m=0
    coef_cos = g20
    Br += pow_rho * 3 * coef_cos * P[3]
    Bt -= pow_rho * coef_cos * dP[3]
    # m=1
    coef_cos = g21 * c_phi + h21 * s_phi
    coef_sin = g21 * s_phi - h21 * c_phi
    Br += pow_rho * 3 * coef_cos * P[4]
    Bt -= pow_rho * coef_cos * dP[4]
    if sin_th > 1e-10: Bp -= pow_rho * 1 * coef_sin * P[4] / sin_th
    # m=2
    c2 = c_phi*c_phi - s_phi*s_phi; s2 = 2*s_phi*c_phi
    coef_cos = g22 * c2 + h22 * s2
    coef_sin = g22 * s2 - h22 * c2
    Br += pow_rho * 3 * coef_cos * P[5]
    Bt -= pow_rho * coef_cos * dP[5]
    if sin_th > 1e-10: Bp -= pow_rho * 2 * coef_sin * P[5] / sin_th

    # Similar for n=3, 4 ... 
    # For now, let's keep the existing loop logic but use the flat P/dP and optimized coeffs
    g_coeffs = np.array([0, g10, g11, g20, g21, g22, g30, g31, g32, g33, g40, g41, g42, g43, g44])
    h_coeffs = np.array([0, 0, h11, 0, h21, h22, 0, h31, h32, h33, 0, h41, h42, h43, h44])
    
    for n in range(1, 5):
        pow_rho = rho**(n + 2)
        for m in range(n + 1):
            idx = n*(n+1)//2 + m
            if m == 0:
                c_m = 1.0; s_m = 0.0
            elif m == 1:
                c_m = c_phi; s_m = s_phi
            else:
                c_m = np.cos(m*phi); s_m = np.sin(m*phi)
                
            coef_cos = g_coeffs[idx] * c_m + h_coeffs[idx] * s_m
            coef_sin = g_coeffs[idx] * s_m - h_coeffs[idx] * c_m
            
            Br += pow_rho * (n + 1) * coef_cos * P[idx]
            Bt -= pow_rho * coef_cos * dP[idx]
            if sin_th > 1e-10:
                Bp -= pow_rho * m * coef_sin * P[idx] / sin_th

    Br *= 1e-9; Bt *= 1e-9; Bp *= 1e-9
    Bx_ecef = Br * sin_th * np.cos(phi) + Bt * cos_th * np.cos(phi) - Bp * np.sin(phi)
    By_ecef = Br * sin_th * np.sin(phi) + Bt * cos_th * np.sin(phi) + Bp * np.cos(phi)
    Bz_ecef = Br * cos_th - Bt * sin_th

    return np.array([Bx_ecef, By_ecef, Bz_ecef])

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