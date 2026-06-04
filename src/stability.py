import numpy as np
from tqdm import tqdm
from engine import integrate_system
import params as p

def check_state_sanity(X, params):
    """
    Checks if the state vector contains physical anomalies.
    Returns (is_sane, reason)
    """
    if np.any(np.isnan(X)):
        return False, "NaN detected in state vector"
    
    num_masses = params.num_masses
    pos = X[:3*num_masses].reshape((num_masses, 3))
    vel = X[3*num_masses:].reshape((num_masses, 3))
    
    for i in range(num_masses):
        r_mag = np.linalg.norm(pos[i])
        alt = r_mag - params.R_e
        if alt < 100e3:
            return False, f"Mass {i} re-entered atmosphere (alt={alt/1e3:.1f} km)"
        
    l_seg_nom = params.L_edt / params.N_edt
    for i in range(params.N_edt):
        l_seg = np.linalg.norm(pos[i+1] - pos[i])
        if l_seg > l_seg_nom * 1.5:
            return False, f"EDT segment {i} stretched beyond limit ({l_seg:.1f}m > {l_seg_nom*1.5:.1f}m)"

    return True, ""


def get_stiffness_report(params):
    """
    Analyzes system stiffness to warn about potential numerical issues.
    """
    # EDT Segment Stiffness
    l_seg = params.L_edt / params.N_edt
    m_seg = params.m_edt_total / params.N_edt
    area_edt = np.pi * (params.diam_edt / 2.0)**2
    k_seg = (params.E_edt * area_edt) / l_seg
    w_seg = np.sqrt(k_seg / m_seg)
    f_seg = w_seg / (2 * np.pi)
    
    # Rope Stiffness
    area_rope = np.pi * (params.diam_rope / 2.0)**2
    k_rope = (params.E_rope * area_rope) / params.L_rope
    w_rope = np.sqrt(k_rope / params.m_sc)
    f_rope = w_rope / (2 * np.pi)
    
    report = {
        "max_freq_hz": max(f_seg, f_rope),
        "edt_freq_hz": f_seg,
        "rope_freq_hz": f_rope,
        "is_stiff": max(f_seg, f_rope) > 10.0 # Heuristic for LSODA
    }
    return report


def analyze_stress_evolution(t_vals, X_vals, params, breaking_tension):
    """
    Analyzes peak tension and checks for oscillation growth (instability).
    """
    num_masses = params.num_masses
    l0_seg = params.L_edt / params.N_edt
    area_edt = np.pi * (params.diam_edt / 2.0)**2
    k_seg = (params.E_edt * area_edt) / l0_seg
    beta = params.beta_edt

    window_count = 4
    window_size = len(t_vals) // window_count
    peaks = []
    
    for w in range(window_count):
        start, end = w * window_size, (w + 1) * window_size
        win_max = 0.0
        for i in range(start, end):
            pos = X_vals[i, :3*num_masses].reshape((num_masses, 3))
            vel = X_vals[i, 3*num_masses:].reshape((num_masses, 3))
            
            for j in range(params.N_edt):
                r_seg = pos[j+1] - pos[j]
                v_seg = vel[j+1] - vel[j]
                l_seg = np.linalg.norm(r_seg)
                dl = l_seg - l0_seg
                l_dot = np.dot(r_seg, v_seg) / max(l_seg, 1e-6)
                
                exp_arg = np.clip(-50.0 * dl, -500, 500)
                scale = 1.0 / (1.0 + np.exp(exp_arg))
                tension = (k_seg * dl + (beta * k_seg) * l_dot) * scale
                win_max = max(win_max, tension)
        peaks.append(win_max)

    overall_max_tension = max(peaks)
    sf = breaking_tension / max(overall_max_tension, 1e-9) # safety factor
    is_growing = peaks[-1] > peaks[1] * 1.2 if len(peaks) > 1 else False
    
    return overall_max_tension, sf, is_growing


def run_preflight_stability_check(X0, p_arr, params, method, duration=300.0):
    """
    Runs a 300s WORST-CASE Stress Test to catch environmental instabilities.
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
        pos[i] += 0.005 * (target_pos - pos[i]) # 0.5% slack
    
    kick_dir = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
    vel[0] += kick_dir * 1.0 
    
    X_stressed[:3*num_masses] = pos.flatten()
    X_stressed[3*num_masses:] = vel.flatten()
    
    p_stressed = p_arr.copy()
    p_stressed[p.IDX_R_LOAD] = 10.0 # Extreme current
    p_stressed[p.IDX_CD] = 4.0     # Extreme drag
    
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
