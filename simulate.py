import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import SimulationParams
from dynamics import tether_dynamics_fast, get_mass_fast

def simulate():
    params = SimulationParams()
    p_arr = params.to_numba_params() # Convert to flat array for Numba
    
    # 1. Initial State Setup
    a_init = params.R_e + params.alt
    v_orb = np.sqrt(params.mu / a_init)
    omega = v_orb / a_init
    
    num_masses = params.num_masses
    X0 = np.zeros(6 * num_masses)
    
    # Position indices (ECI)
    # Configuration: Tip (lowest) -> EDT -> SC -> Target (highest, at ref altitude)
    r_target = np.array([a_init, 0.0, 0.0])
    v_target = np.array([0.0, v_orb, 0.0])
    
    pos = np.zeros((num_masses, 3))
    vel = np.zeros((num_masses, 3))
    
    # Target is at the highest point
    # Tip (Index 0)
    dist_tip = params.L_edt + params.L_rope
    pos[0] = r_target - np.array([dist_tip, 0.0, 0.0])
    vel[0] = v_target - np.array([0.0, omega * dist_tip, 0.0])
    
    # EDT beads (Index 1 to N_edt)
    L0_seg = params.L_edt / params.N_edt
    for i in range(1, params.N_edt + 1):
        dist = params.L_rope + (params.N_edt - i + 1) * L0_seg
        pos[i] = r_target - np.array([dist, 0.0, 0.0])
        vel[i] = v_target - np.array([0.0, omega * dist, 0.0])
        
    # SC (Index N_edt + 1)
    pos[params.N_edt + 1] = r_target - np.array([params.L_rope, 0.0, 0.0])
    vel[params.N_edt + 1] = v_target - np.array([0.0, omega * params.L_rope, 0.0])
    
    # Target (Index N_edt + 2)
    pos[params.N_edt + 2] = r_target
    vel[params.N_edt + 2] = v_target
        
    X0[:3*num_masses] = pos.flatten()
    X0[3*num_masses:] = vel.flatten()
    
    # 2. Integration
    t_span = (0, 5400 * 2) # 2 Orbits

    from tqdm import tqdm
    pbar_container = [None]
    last_t_rounded = [0]

    def wrapped_dynamics(t, y):
        if pbar_container[0] is None:
            pbar_container[0] = tqdm(total=int(t_span[1]), unit=' seconds', desc="Propagating Orbit")
        t_now_rounded = int(t)
        if t_now_rounded > last_t_rounded[0]:
            pbar_container[0].update(t_now_rounded - last_t_rounded[0])
            last_t_rounded[0] = t_now_rounded
        return tether_dynamics_fast(t, y, p_arr)

    print("Starting simulation (Python + Numba JIT)...")
    sol = solve_ivp(
        wrapped_dynamics,
        t_span,
        X0,
        method='LSODA',
        rtol=1e-4,
        atol=1e-6
    )
    if pbar_container[0] is not None:
        pbar_container[0].close()
    print(f"Simulation finished. Status: {sol.message}")
    
    # 3. Post-Processing: SYSTEM CENTER OF MASS SMA
    t_vals = sol.t
    X_vals = sol.y.T
    
    # Calculate System CoM and SMA
    total_m = 0
    masses = []
    for i in range(num_masses):
        m = get_mass_fast(i, p_arr, num_masses)
        masses.append(m)
        total_m += m
    
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
    
    import pandas as pd
    import os
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cols = ['time_s', 'sma_com_km']
    for i in range(num_masses):
        label = f"m{i}_target" if i == params.N_edt + 2 else (f"m{i}_sc" if i == params.N_edt + 1 else (f"m{i}_tip" if i == 0 else f"m{i}_bead"))
        cols += [f'{label}_rx_m', f'{label}_ry_m', f'{label}_rz_m', f'{label}_vx_ms', f'{label}_vy_ms', f'{label}_vz_ms']
        
    data_out = np.hstack([t_vals.reshape(-1, 1), sma_com.reshape(-1, 1), X_vals])
    df = pd.DataFrame(data_out, columns=cols)
    csv_filename = os.path.join(results_dir, "simulation_results.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Data exported successfully to {csv_filename}")

    # Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_vals/60, (sma_com - sma_com[0])/1e3)
    plt.grid(True)
    plt.xlabel('Time [minutes]')
    plt.ylabel('Δ System SMA (CoM) [km]')
    plt.title('System Orbital Decay (Stable CoM)')
    
    plt.subplot(1, 2, 2)
    final_pos = X_vals[-1, :3*num_masses].reshape((num_masses, 3))
    pos_target = final_pos[params.N_edt + 2]
    rel_pos = final_pos - pos_target
    plt.plot(rel_pos[:, 0], rel_pos[:, 1], '-ok')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.xlabel('Radial [m]')
    plt.ylabel('In-Track [m]')
    plt.title('Final Tether Configuration')
    
    plt.tight_layout()
    plot_filename = os.path.join(results_dir, "simulation_plots.png")
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    simulate()
