import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import njit
from dynamics import get_mass_fast, compute_physics_metrics, smooth_tension
import params as p
import yaml
import hashlib

@njit
def calculate_total_energy_fast(p_frame, v_frame, masses, p_arr):
    """
    Numba-optimized energy calculation for a single frame.
    
    Mathematical Physics:
    1. Kinetic Energy: $T = \sum 0.5 m_i v_i^2$
    2. Gravitational Potential: $U_g = \sum -G M m_i / r_i$
    3. Elastic Potential (EDT + Rope): $U_e = \sum 0.5 k (\Delta L)^2$ for $L > L_0$, else 0.
    """
    num_masses = len(masses)
    mu = p_arr[p.IDX_MU]
    n_edt = int(p_arr[p.IDX_N_EDT])
    is_sc_edt_target = p_arr[p.IDX_SYSTEM_CONFIG] > 0.5
    
    # Derived EDT Stiffness
    area_edt = p_arr[p.IDX_AREA_EDT]
    l_edt_seg = p_arr[p.IDX_L_EDT] / n_edt
    k_edt = (p_arr[p.IDX_E_EDT] * area_edt) / l_edt_seg
    
    # Derived Rope Stiffness
    k_rope = p_arr[p.IDX_K_ROPE]
    l_rope_nom = p_arr[p.IDX_L_ROPE]
    
    e_total = 0.0
    for j in range(num_masses):
        r_mag = np.linalg.norm(p_frame[j])
        v_mag2 = np.sum(v_frame[j]**2)
        e_total += 0.5 * masses[j] * v_mag2 - (mu * masses[j] / r_mag)
        
    for j in range(n_edt):
        dr = np.linalg.norm(p_frame[j+1] - p_frame[j])
        if dr > l_edt_seg:
            e_total += 0.5 * k_edt * (dr - l_edt_seg)**2
            
    if not is_sc_edt_target:
        dr_rope = np.linalg.norm(p_frame[n_edt+1] - p_frame[n_edt])
        if dr_rope > l_rope_nom:
            e_total += 0.5 * k_rope * (dr_rope - l_rope_nom)**2
    return e_total

def calculate_com_sma(t_vals, X_vals, p_arr, params):
    """
    Calculates the Semi-Major Axis (SMA) of the system's Center of Mass (CoM).
    
    Mathematical Realism:
    In a tethered system, calculating SMA for an individual body is misleading due to 
    large velocity oscillations caused by libration. 
    1. CoM Position: $R_{com} = \frac{\sum m_i r_i}{\sum m_i}$
    2. CoM Velocity: $V_{com} = \frac{\sum m_i v_i}{\sum m_i}$
    3. Specific Orbital Energy: $\epsilon = \frac{V_{com}^2}{2} - \frac{\mu}{|R_{com}|}$
    4. SMA: $a = -\frac{\mu}{2\epsilon}$
    
    This provides a stable representation of the system's global orbital decay 
    by averaging out the internal multi-body dynamics.
    """
    num_masses = params.num_masses
    masses = [get_mass_fast(i, p_arr, num_masses) for i in range(num_masses)]
    total_m = sum(masses)
    
    com_pos = np.zeros((len(t_vals), 3))
    com_vel = np.zeros((len(t_vals), 3))
    
    for i in range(num_masses):
        m = masses[i]
        com_pos += (m / total_m) * X_vals[:, 3*i : 3*i+3]
        com_vel += (m / total_m) * X_vals[:, 3*num_masses + 3*i : 3*num_masses + 3*i+3]
    
    r_com = np.linalg.norm(com_pos, axis=1)
    v_com2 = np.sum(com_vel**2, axis=1)
    energy_com = v_com2/2.0 - params.mu / r_com
    sma_com = -params.mu / (2.0 * energy_com)
    return sma_com

def post_process_telemetry(t_vals, X_vals, p_arr, params, include_sma=True, sma_array=None):
    """
    Extracts high-fidelity physical metrics from the integrated state vector.
    
    Metrics Calculated:
    1. Mechanical Energy: Validates the integrity of the conservative physics.
    2. Libration (Pitch): Angle between radial and tether axis.
    3. SMA: Orbital decay tracking.
    4. EDT Current: Dynamic current from motional EMF.
    5. Forces: Lorentz vs. Drag magnitude comparison.
    """
    num_masses = params.num_masses
    n_edt = params.N_edt
    masses = np.array([get_mass_fast(i, p_arr, num_masses) for i in range(num_masses)])
    total_m = np.sum(masses)
    is_sc_edt_target = (params.system_config == 'SC_EDT_TARGET')
    
    count = len(t_vals)
    energy = np.zeros(count)
    rope_L = np.zeros(count)
    edt_L = np.zeros(count)
    pitch = np.zeros(count)
    current = np.zeros(count)
    lorentz = np.zeros(count)
    drag = np.zeros(count)
    
    sma = np.zeros(count)
    if include_sma:
        if sma_array is not None:
            sma = sma_array
        else:
            sma = calculate_com_sma(t_vals, X_vals, p_arr, params)

    if is_sc_edt_target:
        idx_start = 0 # SC
        idx_end = n_edt # Target
    else:
        idx_start = 0 # Tip
        idx_sc = n_edt
        idx_end = n_edt + 1

    # Initialize telemetry dict
    telemetry = {
        "sma_km": sma / 1000.0,
        "current_a": current,
        "lorentz_n": lorentz,
        "drag_n": drag,
        "pitch_deg": pitch,
        "energy_j": energy,
        "rope_l_m": rope_L,
        "edt_l_m": edt_L
    }

    # Derived EDT Stiffness
    area_edt = p_arr[p.IDX_AREA_EDT]
    l_edt_seg = p_arr[p.IDX_L_EDT] / n_edt
    k_edt_seg = (p_arr[p.IDX_E_EDT] * area_edt) / l_edt_seg
    beta_edt = p_arr[p.IDX_BETA_EDT]

    for i in range(count):
        p_frame = X_vals[i, :3*num_masses].reshape((num_masses, 3))
        v_frame = X_vals[i, 3*num_masses:].reshape((num_masses, 3))
        
        energy[i] = calculate_total_energy_fast(p_frame, v_frame, masses, p_arr)
        
        # Calculate Tension per segment
        for j in range(n_edt):
            p_a, p_b = p_frame[j], p_frame[j+1]
            v_a, v_b = v_frame[j], v_frame[j+1]
            
            r_seg = p_b - p_a
            v_seg = v_b - v_a
            l_seg = np.linalg.norm(r_seg)
            l_seg_safe = max(l_seg, 1e-6)
            l_dot_seg = np.dot(r_seg, v_seg) / l_seg_safe
            
            t_seg = smooth_tension(l_seg - l_edt_seg, l_dot_seg, k_edt_seg, beta_edt)
            
            # Store in the telemetry dict
            key = f"edt_tension_{j}_n"
            if key not in telemetry:
                telemetry[key] = np.zeros(count)
            telemetry[key][i] = t_seg

        if is_sc_edt_target:
            rope_L[i] = 0.0
            edt_L[i] = np.linalg.norm(p_frame[idx_end] - p_frame[idx_start])
        else:
            rope_L[i] = np.linalg.norm(p_frame[idx_end] - p_frame[idx_sc])
            edt_L[i] = np.linalg.norm(p_frame[idx_sc] - p_frame[idx_start])
        
        # Libration
        r_com = np.zeros(3)
        for j in range(num_masses):
            r_com += (masses[j] / total_m) * p_frame[j]
        r_com_mag = np.linalg.norm(r_com)
        u_v = r_com / max(r_com_mag, 1e-6)
        
        tether_vec = p_frame[idx_end] - p_frame[idx_start]
        tether_len = np.linalg.norm(tether_vec)
        u_tether = tether_vec / max(tether_len, 1e-6)
        
        pitch[i] = np.degrees(np.arccos(np.clip(np.dot(u_v, u_tether), -1.0, 1.0)))
        
        # Physics Metrics (Current, Forces)
        curr, lor, drg = compute_physics_metrics(t_vals[i], X_vals[i], p_arr)
        current[i] = curr
        lorentz[i] = lor
        drag[i] = drg
        
    return telemetry

def save_csv(filename, run_folder, t_vals, telemetry_dict, X_vals, params, silent=False, append=False):
    """
    Saves simulation results to a standardized CSV format.
    telemetry_dict: Dictionary of arrays to be saved as columns.
    """
    num_masses = params.num_masses
    n_edt = params.N_edt
    is_sc_edt_target = (params.system_config == 'SC_EDT_TARGET')
    
    tel_names = list(telemetry_dict.keys())
    tel_data = np.column_stack([telemetry_dict[name] for name in tel_names])
    cols = ['time_s'] + tel_names

    for i in range(num_masses):
        if is_sc_edt_target:
            label = f"m{i}_target" if i == n_edt else (f"m{i}_sc" if i == 0 else f"m{i}_bead")
        else:
            label = f"m{i}_target" if i == n_edt + 1 else (f"m{i}_sc" if i == n_edt else (f"m{i}_tip" if i == 0 else f"m{i}_bead"))
        cols += [f'{label}_rx_m', f'{label}_ry_m', f'{label}_rz_m', f'{label}_vx_ms', f'{label}_vy_ms', f'{label}_vz_ms']

    pos_data = X_vals[:, :3*num_masses].reshape(-1, num_masses, 3)
    vel_data = X_vals[:, 3*num_masses:].reshape(-1, num_masses, 3)

    interleaved_data = np.zeros((len(t_vals), 6*num_masses))
    for i in range(num_masses):
        interleaved_data[:, 6*i:6*i+3] = pos_data[:, i, :]
        interleaved_data[:, 6*i+3:6*i+6] = vel_data[:, i, :]

    filepath = os.path.join(run_folder, filename)
    data_out = np.hstack([t_vals.reshape(-1, 1), tel_data, interleaved_data])
    df = pd.DataFrame(data_out, columns=cols)

    mode = 'a' if append else 'w'
    header = not append or not os.path.exists(filepath)
    df.to_csv(filepath, index=False, mode=mode, header=header)
    if not silent:
        print(f"Data {'appended' if append else 'saved'} to {filepath}")

def calculate_mission_results(t_vals, sma_com, params, total_compute_time=0.0):
    """
    Calculates comprehensive mission metrics from simulation history.
    Uses linear regression to find the mean decay rate, filtering out 
    oscillations from J2 and libration.

    Units: Meters, Seconds.
    """
    if sma_com is None or len(sma_com) < 2:
        return {}

    # 1. Linear Regression for Statistical Mean Decay Rate
    # SMA(t) = intercept + slope * t
    # Slope is the mean decay rate in m/s.
    t_clean = t_vals - t_vals[0]
    n = len(t_clean)

    # Simple Least Squares
    sum_t = np.sum(t_clean)
    sum_t2 = np.sum(t_clean**2)
    sum_s = np.sum(sma_com)
    sum_ts = np.sum(t_clean * sma_com)

    denom = (n * sum_t2 - sum_t**2)
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sum_ts - sum_t * sum_s) / denom

    # We define decay rate as positive for losing altitude
    decay_rate_mps = -slope 
    initial_sma = float(sma_com[0])
    final_sma = float(sma_com[-1])
    sma_drop_total = initial_sma - final_sma
    sim_duration = float(t_vals[-1] - t_vals[0])

    # Orbital Period T = 2 * pi * sqrt(a^3 / mu)
    # Using initial SMA for period consistency
    period_init = 2.0 * np.pi * np.sqrt(initial_sma**3 / params.mu)
    decay_per_orbit_m = decay_rate_mps * period_init
    return {
        "com_sma_initial_m": initial_sma,
        "com_sma_final_m": final_sma,
        "com_sma_drop_total_m": sma_drop_total,
        "mean_decay_rate_mps": float(decay_rate_mps),
        "mean_decay_rate_kmhr": float(decay_rate_mps * 3.6),
        "mean_decay_rate_kmyear": float(decay_rate_mps * 3.6 * 24 * 365.25),
        "mean_decay_per_orbit_m": float(decay_per_orbit_m),
        "initial_orbital_period_s": float(period_init),
        "simulation_time_s": sim_duration,
        "total_simulated_time_s": float(t_vals[-1]),
        "total_compute_time_s": float(total_compute_time)
    }

def save_config_params_results_yaml(filename, run_folder, t_vals, sma_com=None, params=None, p_arr=None, is_final=False, silent=False, total_compute_time=0.0):
    """
    Saves simulation parameters and key results to a YAML file for easy reference.
    Differentiates between initial setup and final mission report.
    """
    run_id = os.path.basename(os.path.normpath(run_folder))

    results = {}
    if sma_com is not None:
        results = calculate_mission_results(t_vals, sma_com, params, total_compute_time)

    # Generate Hashes
    p_hash = "N/A"
    report_hash = "N/A"

    if p_arr is not None:
        p_hash = hashlib.sha256(p_arr.tobytes()).hexdigest()
        if is_final and results:
            # Generate a composite hash for the entire report (Params + Results)
            # KISS: Join p_hash with a string representation of the results dict
            report_input = p_hash + str(results)
            report_hash = hashlib.sha256(report_input.encode()).hexdigest()

    output_data = {
        "metadata": {"run_id": run_id, "description": "EDT Orbital Decay Simulation", "parameter_hash": p_hash, "report_hash": report_hash},
        "parameters": params.to_dict() if params else {},
    }
    if results: output_data["results"] = results
    filepath = os.path.join(run_folder, filename)
    with open(filepath, 'w') as file:
        yaml.dump(output_data, file, default_flow_style=False, sort_keys=False)
    if not silent: print(f"{'Final mission report' if is_final else 'Simulation parameters initialized'} at {filepath}")
