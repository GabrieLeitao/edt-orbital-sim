import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import SimulationParams
import params as p
from dynamics import tether_dynamics_fast, get_mass_fast
from numba import njit
from tqdm import tqdm
import os

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
    
    # 1. Kinetic and Gravitational Potential
    for j in range(num_masses):
        r_mag = np.linalg.norm(p_frame[j])
        v_mag2 = np.sum(v_frame[j]**2)
        e_total += 0.5 * masses[j] * v_mag2 - (mu * masses[j] / r_mag)
        
    # 2. Elastic Potential Energy
    # EDT Segments
    for j in range(n_edt + 1):
        dr = np.linalg.norm(p_frame[j+1] - p_frame[j])
        if dr > l_edt_seg:
            e_total += 0.5 * k_edt * (dr - l_edt_seg)**2
            
    # Rope Segment
    dr_rope = np.linalg.norm(p_frame[n_edt+2] - p_frame[n_edt+1])
    if dr_rope > l_rope_nom:
        e_total += 0.5 * k_rope * (dr_rope - l_rope_nom)**2
        
    return e_total

def validate_conservation():
    """
    Runs a simulation with ZERO current and ZERO drag to verify 
    Mechanical Energy Conservation of the LMM dynamics engine.
    """
    params = SimulationParams()
    params.I_edt = 0.0
    params.Cd = 0.0
    p_arr = params.to_numba_params()
    num_masses = params.num_masses
    
    # 1. Setup Initial State
    a_init = params.R_e + params.alt
    v_orb = np.sqrt(params.mu / a_init)
    omega = v_orb / a_init
    X0 = np.zeros(6 * num_masses)
    pos = np.zeros((num_masses, 3))
    vel = np.zeros((num_masses, 3))
    r_target = np.array([a_init, 0.0, 0.0])
    v_target = np.array([0.0, v_orb, 0.0])
    dist_tip = params.L_edt + params.L_rope
    pos[0] = r_target - np.array([dist_tip, 0.0, 0.0])
    vel[0] = v_target - np.array([0.0, omega * dist_tip, 0.0])
    L0_seg = params.L_edt / params.N_edt
    for i in range(1, params.N_edt + 1):
        dist = params.L_rope + (params.N_edt - i + 1) * L0_seg
        pos[i] = r_target - np.array([dist, 0.0, 0.0])
        vel[i] = v_target - np.array([0.0, omega * dist, 0.0])
    pos[params.N_edt + 1] = r_target - np.array([params.L_rope, 0.0, 0.0])
    vel[params.N_edt + 1] = v_target - np.array([0.0, omega * params.L_rope, 0.0])
    pos[params.N_edt + 2] = r_target
    vel[params.N_edt + 2] = v_target
    X0[:3*num_masses] = pos.flatten()
    X0[3*num_masses:] = vel.flatten()

    # 2. Integration with Progress Bar
    t_span = (0, 5400) 
    pbar_container = [None]
    last_t_rounded = [0]

    def wrapped_dynamics(t, y):
        if pbar_container[0] is None:
            pbar_container[0] = tqdm(total=int(t_span[1]), unit=' seconds', desc="Validating Physics (Conservative)")
        t_now_rounded = int(t)
        if t_now_rounded > last_t_rounded[0]:
            pbar_container[0].update(t_now_rounded - last_t_rounded[0])
            last_t_rounded[0] = t_now_rounded
        return tether_dynamics_fast(t, y, p_arr)

    print("Starting Validation (Python + Numba JIT)...")
    sol = solve_ivp(
        wrapped_dynamics,
        t_span, X0, method='LSODA', rtol=1e-7, atol=1e-9
    )
    if pbar_container[0] is not None:
        pbar_container[0].close()
    
    # 3. Fast Energy Calculation
    print("Post-processing: Calculating energy conservation metrics...")
    t_vals = sol.t
    X_vals = sol.y.T
    energy_total = np.zeros(len(t_vals))
    masses = np.array([get_mass_fast(i, p_arr, num_masses) for i in range(num_masses)])
    
    for i in range(len(t_vals)):
        p_frame = X_vals[i, :3*num_masses].reshape((num_masses, 3))
        v_frame = X_vals[i, 3*num_masses:].reshape((num_masses, 3))
        energy_total[i] = calculate_total_energy_fast(p_frame, v_frame, masses, p_arr)

    # 4. Geometric & Kinematic Validation
    print("Post-processing: Checking geometric constraints and libration...")
    rope_lengths = np.zeros(len(t_vals))
    edt_lengths = np.zeros(len(t_vals))
    pitch_angles = np.zeros(len(t_vals)) 

    idx_sc = params.N_edt + 1
    idx_target = params.N_edt + 2

    for i in range(len(t_vals)):
        p_frame = X_vals[i, :3*num_masses].reshape((num_masses, 3))
        rope_vec = p_frame[idx_target] - p_frame[idx_sc]
        rope_lengths[i] = np.linalg.norm(rope_vec)
        edt_vec = p_frame[idx_sc] - p_frame[0]
        edt_lengths[i] = np.linalg.norm(edt_vec)
        r_com = np.zeros(3)
        total_m = np.sum(masses)
        for j in range(num_masses):
            r_com += (masses[j] / total_m) * p_frame[j]
        u_v = r_com / np.linalg.norm(r_com)
        u_tether = (p_frame[idx_target] - p_frame[0]) / (rope_lengths[i] + edt_lengths[i])
        pitch_angles[i] = np.degrees(np.arccos(np.clip(np.dot(u_v, u_tether), -1.0, 1.0)))

    # 5. Plot Results
    rel_energy_error = (energy_total - energy_total[0]) / np.abs(energy_total[0])
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(t_vals, rel_energy_error)
    plt.grid(True)
    plt.title("Energy Conservation Error (Target: < 1e-4)")
    plt.ylabel("(E-E0)/|E0|")
    plt.subplot(2, 2, 2)
    plt.plot(t_vals, rope_lengths, label='Rope (50m)')
    plt.axhline(y=params.L_rope, color='r', linestyle='--')
    plt.grid(True)
    plt.title("Rope Geometric Constraint")
    plt.ylabel("Length [m]")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(t_vals, pitch_angles)
    plt.grid(True)
    plt.title("Libration Stability (In-Plane Pitch)")
    plt.ylabel("Angle [deg]")
    plt.xlabel("Time [s]")
    plt.subplot(2, 2, 4)
    plt.plot(t_vals, edt_lengths)
    plt.axhline(y=params.L_edt, color='r', linestyle='--')
    plt.grid(True)
    plt.title("EDT Structural Integrity")
    plt.ylabel("Total EDT Length [m]")
    plt.xlabel("Time [s]")
    
    plt.tight_layout()
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_filename = os.path.join(results_dir, "validation_plots.png")
    plt.savefig(plot_filename)
    print(f"Validation plots saved to {plot_filename}")
    plt.show()
    
    max_err = np.max(np.abs(rel_energy_error))
    print(f"\n--- Validation Report ---")
    print(f"1. Energy Stability: {max_err:.2e}")
    print(f"2. Max Rope Stretch: {np.max(rope_lengths) - params.L_rope:.4f} m")
    print(f"3. Max Pitch Libration: {np.max(pitch_angles):.2f} degrees")
    print(f"4. Structure Check: {'Stable' if np.max(rope_lengths) < params.L_rope * 1.1 else 'UNSTABLE'}")
    
    import pandas as pd
    cols = ['time_s', 'rel_energy_error']
    for i in range(num_masses):
        label = f"m{i}_target" if i == params.N_edt + 2 else (f"m{i}_sc" if i == params.N_edt + 1 else (f"m{i}_tip" if i == 0 else f"m{i}_bead"))
        cols += [f'{label}_rx_m', f'{label}_ry_m', f'{label}_rz_m', f'{label}_vx_ms', f'{label}_vy_ms', f'{label}_vz_ms']
    data_out = np.hstack([t_vals.reshape(-1, 1), rel_energy_error.reshape(-1, 1), X_vals])
    val_df = pd.DataFrame(data_out, columns=cols)
    val_csv = os.path.join(results_dir, "validation_results.csv")
    val_df.to_csv(val_csv, index=False)
    print(f"Full validation state exported to {val_csv}")

    if max_err < 1e-4:
        print("PASS: Physics engine conserves energy correctly.")
    else:
        print("FAIL: Significant energy drift detected.")

if __name__ == "__main__":
    validate_conservation()
