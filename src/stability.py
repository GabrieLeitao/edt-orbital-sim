import numpy as np
from tqdm import tqdm
from numba import njit
from engine import integrate_system
import params as p

# ==========================================
# NUMBA ACCELERATED COMPUTATIONAL BACKENDS
# ==========================================

@njit(cache=True)
def _check_state_sanity_fast(X, num_masses, R_e, N_edt, l_seg_nom):
    """
    Fast Numba loop tracking mathematical boundaries.
    Returns: (error_code, index, value)
    Codes: 0=Sane, -1=NaN, 1=Atmosphere Re-entry, 2=Stretched limit
    """
    # 1. NaN Vector Fast Scan
    for i in range(X.shape[0]):
        if np.isnan(X[i]):
            return -1, 0.0, 0.0
    
    # 2. Check Altitude Bounds
    for i in range(num_masses):
        px = X[3 * i]
        py = X[3 * i + 1]
        pz = X[3 * i + 2]
        r_mag = np.sqrt(px*px + py*py + pz*pz)
        alt = r_mag - R_e
        if alt < 100e3:
            return 1, float(i), alt
            
    # 3. Check Segment Stretching Bounds
    for i in range(N_edt):
        p1_idx = 3 * i
        p2_idx = 3 * (i + 1)
        rx = X[p2_idx] - X[p1_idx]
        ry = X[p2_idx + 1] - X[p1_idx + 1]
        rz = X[p2_idx + 2] - X[p1_idx + 2]
        l_seg = np.sqrt(rx*rx + ry*ry + rz*rz)
        if l_seg > l_seg_nom * 1.5:
            return 2, float(i), l_seg
            
    return 0, 0.0, 0.0


@njit(cache=True)
def _analyze_stress_evolution_fast(X_vals, num_masses, N_edt, l0_seg, k_seg, beta, window_size, window_count):
    """
    Highly optimized multi-nested loop executing raw array math.
    Eliminates internal memory resizing and reshaping allocations completely.
    """
    peaks = np.zeros(window_count)
    
    for w in range(window_count):
        start = w * window_size
        end = (w + 1) * window_size
        win_max = 0.0
        
        for i in range(start, end):
            for j in range(N_edt):
                # Flat index mapping for positions
                p1_idx = 3 * j
                p2_idx = 3 * (j + 1)
                # Flat index mapping for velocities
                v1_idx = 3 * num_masses + 3 * j
                v2_idx = 3 * num_masses + 3 * (j + 1)
                
                # Element-wise delta computations (Vectorized-equivalent but allocation-free)
                rx = X_vals[i, p2_idx] - X_vals[i, p1_idx]
                ry = X_vals[i, p2_idx + 1] - X_vals[i, p1_idx + 1]
                rz = X_vals[i, p2_idx + 2] - X_vals[i, p1_idx + 2]
                
                vx = X_vals[i, v2_idx] - X_vals[i, v1_idx]
                vy = X_vals[i, v2_idx + 1] - X_vals[i, v1_idx + 1]
                vz = X_vals[i, v2_idx + 2] - X_vals[i, v1_idx + 2]
                
                l_seg = np.sqrt(rx*rx + ry*ry + rz*rz)
                dl = l_seg - l0_seg
                l_dot = (rx*vx + ry*vy + rz*vz) / max(l_seg, 1e-6)
                
                # Stable analytical evaluation of exponential scaling
                exp_arg = -50.0 * dl
                if exp_arg < -500.0:
                    exp_arg = -500.0
                elif exp_arg > 500.0:
                    exp_arg = 500.0
                    
                scale = 1.0 / (1.0 + np.exp(exp_arg))
                tension = (k_seg * dl + (beta * k_seg) * l_dot) * scale
                
                if tension > win_max:
                    win_max = tension
                    
        peaks[w] = win_max
        
    return peaks


# ==========================================
# PUBLIC INTERFACE WRAPPERS (OBJECT CONTROLLERS)
# ==========================================

def check_state_sanity(X, params):
    """
    Checks if the state vector contains physical anomalies.
    Decodes the quick calculation results back into user-friendly strings.
    """
    num_masses = params.num_masses
    l_seg_nom = params.L_edt / params.N_edt
    
    err_code, idx, val = _check_state_sanity_fast(X, num_masses, params.R_e, params.N_edt, l_seg_nom)
    
    if err_code == -1:
        return False, "NaN detected in state vector"
    elif err_code == 1:
        return False, f"Mass {int(idx)} re-entered atmosphere (alt={val/1e3:.1f} km)"
    elif err_code == 2:
        return False, f"EDT segment {int(idx)} stretched beyond limit ({val:.1f}m > {l_seg_nom*1.5:.1f}m)"
        
    return True, ""


def get_stiffness_report(params):
    """
    Analyzes system stiffness to warn about potential numerical issues.
    (Kept in Python as it executes lightweight scalar math only once).
    """
    l_seg = params.L_edt / params.N_edt
    m_seg = params.m_edt_total / params.N_edt
    area_edt = np.pi * (params.diam_edt / 2.0)**2
    k_seg = (params.E_edt * area_edt) / l_seg
    w_seg = np.sqrt(k_seg / m_seg)
    f_seg = w_seg / (2 * np.pi)
    
    area_rope = np.pi * (params.diam_rope / 2.0)**2
    k_rope = (params.E_rope * area_rope) / params.L_rope
    w_rope = np.sqrt(k_rope / params.m_sc)
    f_rope = w_rope / (2 * np.pi)
    
    report = {
        "max_freq_hz": max(f_seg, f_rope),
        "edt_freq_hz": f_seg,
        "rope_freq_hz": f_rope,
        "is_stiff": max(f_seg, f_rope) > 10.0
    }
    return report


def analyze_stress_evolution(t_vals, X_vals, params, breaking_tension):
    """
    Analyzes peak tension and checks for oscillation growth via compiled backend.
    """
    num_masses = params.num_masses
    l0_seg = params.L_edt / params.N_edt
    area_edt = np.pi * (params.diam_edt / 2.0)**2
    k_seg = (params.E_edt * area_edt) / l0_seg
    beta = params.beta_edt

    window_count = 4
    window_size = len(t_vals) // window_count
    
    # Delegate the heavy iterations to the Numba layer
    peaks = _analyze_stress_evolution_fast(
        X_vals, num_masses, params.N_edt, l0_seg, k_seg, beta, window_size, window_count
    )

    overall_max_tension = max(peaks)
    sf = breaking_tension / max(overall_max_tension, 1e-9)
    is_growing = peaks[-1] > peaks[1] * 1.2 if len(peaks) > 1 else False
    
    return overall_max_tension, sf, is_growing


def run_preflight_stability_check(X0, p_arr, params, method, duration=300.0):
    """
    Runs a 300s WORST-CASE Stress Test to catch environmental instabilities.
    (Kept in Python due to dependencies on tqdm progress bars and solver instances).
    """
    print(f"--- Pre-flight Worst-Case Stress Test ({duration}s) ---")

    stiffness = get_stiffness_report(params)
    if stiffness['is_stiff']:
        component = "EDT segments" if stiffness['edt_freq_hz'] > stiffness['rope_freq_hz'] else "Rope"
        print(f"WARN: High frequency ({stiffness['max_freq_hz']:.2f} Hz) in {component}. Simulation may be slow.")
    
    X_stressed = X0.copy()
    num_masses = params.num_masses
    pos = X_stressed[:3*num_masses].reshape((num_masses, 3))
    vel = X_stressed[3*num_masses:].reshape((num_masses, 3))
    target_pos = pos[-1]
    for i in range(num_masses - 1):
        pos[i] += 0.005 * (target_pos - pos[i])
    
    kick_dir = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
    vel[0] += kick_dir * 1.0 
    
    X_stressed[:3*num_masses] = pos.flatten()
    X_stressed[3*num_masses:] = vel.flatten()
    
    p_stressed = p_arr.copy()
    p_stressed[p.IDX_R_LOAD] = 10.0 
    p_stressed[p.IDX_CD] = 4.0     
    
    try:
        with tqdm(total=int(duration), unit='s', desc="Worst-Case Test") as pbar:
            sol = integrate_system(X_stressed, (0, duration), p_stressed,
                                   pbar=pbar, sampling_hz=20.0, method=method)
    except Exception as e:
        return False, f"Solver crashed during stress test: {e}"
    
    is_sane, reason = check_state_sanity(sol.y[:, -1], params)
    if not is_sane:
        return False, f"Dynamic Failure: {reason}"
    
    area_edt = np.pi * (params.diam_edt / 2.0)**2
    breaking_t = 270e6 * area_edt
    max_t, sf, growing = analyze_stress_evolution(sol.t, sol.y.T, params, breaking_t)
    
    print(f"Peak Stress Test Tension: {max_t:.2f} N (Safety Factor: {sf:.2f}x)")
    
    if sf < 1.2: 
        return False, f"Tether SNAPPED in worst-case test! Safety Factor={sf:.2f} too low."
    
    if growing and max_t > 5:
        return False, "Unstable Growth detected under worst-case electrical load."
    
    print("Stability Check PASSED.\n")
    return True, "Stable"