import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import njit
from dynamics import get_mass_fast
import params as p

@njit
def calculate_total_energy_fast(p_frame, v_frame, masses, p_arr):
    """Numba-optimized energy calculation for a single frame"""
    num_masses = len(masses)
    mu = p_arr[p.IDX_MU]
    k_edt = p_arr[p.IDX_K_EDT]
    k_rope = p_arr[p.IDX_K_ROPE]
    l_edt_seg = p_arr[p.IDX_L_EDT] / int(p_arr[p.IDX_N_EDT])
    l_rope_nom = p_arr[p.IDX_L_ROPE]
    n_edt = int(p_arr[p.IDX_N_EDT])
    
    e_total = 0.0
    for j in range(num_masses):
        r_mag = np.linalg.norm(p_frame[j])
        v_mag2 = np.sum(v_frame[j]**2)
        e_total += 0.5 * masses[j] * v_mag2 - (mu * masses[j] / r_mag)
        
    for j in range(n_edt + 1):
        dr = np.linalg.norm(p_frame[j+1] - p_frame[j])
        if dr > l_edt_seg:
            e_total += 0.5 * k_edt * (dr - l_edt_seg)**2
            
    dr_rope = np.linalg.norm(p_frame[n_edt+2] - p_frame[n_edt+1])
    if dr_rope > l_rope_nom:
        e_total += 0.5 * k_rope * (dr_rope - l_rope_nom)**2
    return e_total

def calculate_com_sma(t_vals, X_vals, p_arr, params):
    """Calculate System Center of Mass and Semi-Major Axis"""
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

def post_process_telemetry(t_vals, X_vals, p_arr, params):
    """Extract energy, lengths, and libration metrics"""
    num_masses = params.num_masses
    masses = np.array([get_mass_fast(i, p_arr, num_masses) for i in range(num_masses)])
    total_m = np.sum(masses)
    
    energy = np.zeros(len(t_vals))
    rope_L = np.zeros(len(t_vals))
    edt_L = np.zeros(len(t_vals))
    pitch = np.zeros(len(t_vals))

    idx_sc = params.N_edt + 1
    idx_target = params.N_edt + 2

    for i in range(len(t_vals)):
        p_frame = X_vals[i, :3*num_masses].reshape((num_masses, 3))
        v_frame = X_vals[i, 3*num_masses:].reshape((num_masses, 3))
        
        energy[i] = calculate_total_energy_fast(p_frame, v_frame, masses, p_arr)
        
        r_rope = p_frame[idx_target] - p_frame[idx_sc]
        rope_L[i] = np.linalg.norm(r_rope)
        edt_L[i] = np.linalg.norm(p_frame[idx_sc] - p_frame[0])
        
        r_com = np.zeros(3)
        for j in range(num_masses):
            r_com += (masses[j] / total_m) * p_frame[j]
        u_v = r_com / np.linalg.norm(r_com)
        u_tether = (p_frame[idx_target] - p_frame[0]) / (rope_L[i] + edt_L[i])
        pitch[i] = np.degrees(np.arccos(np.clip(np.dot(u_v, u_tether), -1.0, 1.0)))
        
    return energy, rope_L, edt_L, pitch

def save_csv(filename, t_vals, telemetry_val, telemetry_name, X_vals, params):
    """Save results to CSV with standardized columns"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    num_masses = params.num_masses
    cols = ['time_s', telemetry_name]
    for i in range(num_masses):
        label = f"m{i}_target" if i == params.N_edt + 2 else (f"m{i}_sc" if i == params.N_edt + 1 else (f"m{i}_tip" if i == 0 else f"m{i}_bead"))
        cols += [f'{label}_rx_m', f'{label}_ry_m', f'{label}_rz_m', f'{label}_vx_ms', f'{label}_vy_ms', f'{label}_vz_ms']
    
    data_out = np.hstack([t_vals.reshape(-1, 1), telemetry_val.reshape(-1, 1), X_vals])
    pd.DataFrame(data_out, columns=cols).to_csv(os.path.join(results_dir, filename), index=False)
    print(f"Data saved to {os.path.join(results_dir, filename)}")
