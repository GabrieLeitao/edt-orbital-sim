import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import SimulationParams
import params as p
from dynamics import tether_dynamics_fast, get_mass_fast
from numba import njit
from tqdm import tqdm

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

    # 4. Plot Results
    rel_energy_error = (energy_total - energy_total[0]) / np.abs(energy_total[0])

    plt.figure(figsize=(15, 5))

    # Subplot 1: Energy Conservation
    plt.subplot(1, 2, 1)
    plt.plot(t_vals, rel_energy_error)
    plt.grid(True)
    plt.title("Relative Energy Conservation (Conservative Test)")
    plt.xlabel("Time [seconds]")
    plt.ylabel("(E - E0) / |E0|")

    # Subplot 2: Final Configuration
    plt.subplot(1, 2, 2)
    final_pos = X_vals[-1, :3*num_masses].reshape((num_masses, 3))
    # Relative to Target
    pos_target = final_pos[params.N_edt + 2]
    rel_pos = final_pos - pos_target
    plt.plot(rel_pos[:, 0], rel_pos[:, 1], '-ok')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.xlabel('Radial [m]')
    plt.ylabel('In-Track [m]')
    plt.title('Final Tether State (Conservative)')

    plt.tight_layout()
    plt.show()

    max_err = np.max(np.abs(rel_energy_error))
    print(f"Max Relative Energy Error: {max_err:.2e}")

    # 5. Full State CSV Export for universal analysis
    import pandas as pd
    cols = ['time_s', 'rel_energy_error']
    for i in range(num_masses):
        label = f"m{i}_target" if i == params.N_edt + 2 else (f"m{i}_sc" if i == params.N_edt + 1 else (f"m{i}_tip" if i == 0 else f"m{i}_bead"))
        cols += [f'{label}_rx_m', f'{label}_ry_m', f'{label}_rz_m', f'{label}_vx_ms', f'{label}_vy_ms', f'{label}_vz_ms']

    data_out = np.hstack([
        t_vals.reshape(-1, 1), 
        rel_energy_error.reshape(-1, 1), 
        X_vals
    ])

    val_df = pd.DataFrame(data_out, columns=cols)
    val_csv = "validation_results.csv"
    val_df.to_csv(val_csv, index=False)
    print(f"Full validation state exported to {val_csv}")

    if max_err < 1e-4:

        print("PASS: Physics engine conserves energy correctly.")
    else:
        print("FAIL: Significant energy drift detected.")

if __name__ == "__main__":
    validate_conservation()
