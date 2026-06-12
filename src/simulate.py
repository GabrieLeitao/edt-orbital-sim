import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import questionary
import hashlib
import time
import argparse
import uuid
from tqdm import tqdm

from params import SimulationParams
from engine import setup_initial_state, integrate_system, save_checkpoint, load_checkpoint
from analysis import (calculate_com_sma, save_csv, save_config_params_results_yaml, 
                      post_process_telemetry, calculate_mission_results)
from utils import get_results_folder
from frames import eci_to_lvlh

from stability import run_preflight_stability_check, check_state_sanity

def plot_simulation(t_vals, sma_com, X_vals, params, run_folder):
    """Generate deorbiting plots"""
    fig = plt.figure(figsize=(14, 6))
    
    # 1. SMA Plot
    plt.subplot(1, 2, 1)
    t_min = t_vals / 60.0
    sma_delta_km = (sma_com - sma_com[0]) / 1e3
    
    # Calculate a moving average (window = 1 orbital period ~ 90 mins)
    # This filters out J2 oscillations and libration noise.
    # At 1Hz sampling, 5400 points = 1 orbit.
    window = 5400 
    if len(sma_com) > window:
        sma_mean = pd.Series(sma_delta_km).rolling(window=window, center=True).mean()
        plt.plot(t_min, sma_mean, 'r', linewidth=2.5, label='Mean Decay [km]')
    
    plt.plot(t_min, sma_delta_km, 'b', alpha=0.3, label='Δ System SMA (CoM) [km]')
    plt.grid(True)
    plt.xlabel('Time [min]')
    plt.ylabel('Δ System SMA (CoM) [km]')
    plt.title('Orbital Decay Trend')
    plt.legend()
    
    # 2. Tether Configuration
    ax_edt = fig.add_subplot(1, 2, 2)

    final_pos = X_vals[-1, :3*params.num_masses].reshape((params.num_masses, 3))
    final_vel = X_vals[-1, 3*params.num_masses:].reshape((params.num_masses, 3))

    if params.system_config == 'SC_EDT_TARGET':
        target_idx = params.N_edt
        sc_idx = 0
        tip_idx = 0
    else:
        target_idx = params.N_edt + 1
        sc_idx = params.N_edt
        tip_idx = 0

    r_ref = final_pos[target_idx]
    v_ref = final_vel[target_idx]
    r_lvlh = eci_to_lvlh(final_pos, v_ref, r_ref)

    it = r_lvlh[:, 0]
    rd = r_lvlh[:, 2]

    plt.plot(it, rd, '-ok')

    ax_edt.set_title(f"Final Tether Configuration ({params.system_config})")
    ax_edt.set_xlabel("In-Track [m]")
    ax_edt.set_ylabel("Radial [m]")
    ax_edt.grid(True)

    line_edt_full, = ax_edt.plot([], [], 'g-', lw=1.5, alpha=0.8)
    marker_tip_edt, = ax_edt.plot([], [], 'mo', markersize=5, label="Tip/SC")
    marker_sc_edt, = ax_edt.plot([], [], 'bo', markersize=5, label="SC")
    marker_target_edt, = ax_edt.plot([], [], 'rs', markersize=7, label="Target")
    
    line_edt_full.set_data(it, rd)
    marker_tip_edt.set_data([it[tip_idx]], [rd[tip_idx]])
    marker_sc_edt.set_data([it[sc_idx]], [rd[sc_idx]])
    marker_target_edt.set_data([it[target_idx]], [rd[target_idx]])
    zoomed_limit_edt = params.L_edt * 1.4
    ax_edt.set_xlim([-zoomed_limit_edt, zoomed_limit_edt])
    ax_edt.set_ylim([zoomed_limit_edt, -zoomed_limit_edt])
    
    plt.tight_layout()
    plot_path = os.path.join(run_folder, "simulation_plots.png")
    plt.savefig(plot_path)
    print(f"Simulation plots saved at {plot_path}")
    # plt.show()

def find_resumable_runs():
    """Finds folders in results/ that have a checkpoint.npz"""
    if not os.path.exists('results'):
        return []
    
    resumable = []
    for d in os.listdir('results'):
        path = os.path.join('results', d)
        if os.path.isdir(path) and 'checkpoint.npz' in os.listdir(path):
            resumable.append(d)
    return sorted(resumable, reverse=True)

def recover_history_from_csv(run_folder, params):
    """Legacy Fallback: load telemetry history from CSV."""
    csv_path = os.path.join(run_folder, "simulation_results.csv")
    if not os.path.exists(csv_path):
        return None, None
        
    df = pd.read_csv(csv_path)
    all_t = df['time_s'].values
    interleaved = df.iloc[:, 2:].values
    n_m = params.num_masses
    pos_rec = np.zeros((len(all_t), 3*n_m))
    vel_rec = np.zeros((len(all_t), 3*n_m))
    for i in range(n_m):
        pos_rec[:, 3*i:3*i+3] = interleaved[:, 6*i:6*i+3]
        vel_rec[:, 3*i:3*i+3] = interleaved[:, 6*i+3:6*i+6]
    all_X = np.hstack([pos_rec, vel_rec])
    return all_t, all_X

def handle_mission_resumption(t_end, sampling_hz, args=None):
    """Manages the logic for resuming an interrupted simulation using memmaps."""
    if not (args and args.resume):
        return None, 0.0, None, None, None, None, 0, 0.0
        
    resumable_runs = find_resumable_runs()
    if not resumable_runs:
        print("No resumable runs found.")
        return None, 0.0, None, None, None, None, 0, 0.0

    run_name = questionary.select("Select run to resume:", choices=resumable_runs).ask()
    if run_name is None:
        sys.exit(0)
    run_folder = os.path.join('results', run_name)
    
    yaml_path = os.path.join(run_folder, "config_params_results.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: YAML configuration missing for {run_name}. Cannot resume safely.")
        return None, 0.0, None, None, None, None, 0, 0.0

    params = SimulationParams.from_yaml(yaml_path)
    p_hash_curr = hashlib.sha256(params.to_numba_params().tobytes()).hexdigest()
    
    t_s, X0, p_flat, p_hash_stored, curr_idx, total_comp = load_checkpoint(run_folder)
    
    if p_hash_stored and p_hash_curr != p_hash_stored:
        print(f"CRITICAL ERROR: Parameter mismatch detected! Aborting.")
        return None, 0.0, None, None, None, None, 0, 0.0

    # Attach to existing memmaps
    path_t = os.path.join(run_folder, "history_t.dat")
    path_X = os.path.join(run_folder, "history_X.dat")
    
    n_total = int(t_end * sampling_hz) + 1
    n_state = len(X0)
    
    if os.path.exists(path_t) and os.path.exists(path_X):
        hist_t = np.memmap(path_t, dtype='float64', mode='r+', shape=(n_total,))
        hist_X = np.memmap(path_X, dtype='float64', mode='r+', shape=(n_total, n_state))
    else:
        # Fallback to CSV if memmaps missing
        print("Memmaps missing. Recovering from CSV...")
        h_t, h_X = recover_history_from_csv(run_folder, params)
        hist_t = np.memmap(path_t, dtype='float64', mode='w+', shape=(n_total,))
        hist_X = np.memmap(path_X, dtype='float64', mode='w+', shape=(n_total, n_state))
        if h_t is not None:
            hist_t[:len(h_t)] = h_t
            hist_X[:len(h_X)] = h_X
            curr_idx = len(h_t) - 1

    print(f"Resuming mission from t = {t_s:.1f}s (Index {curr_idx})")
    return run_folder, t_s, X0, params, hist_t, hist_X, curr_idx, total_comp

def initialize_new_mission(t_end, sampling_hz, args=None):
    """Sets up a fresh simulation run with pre-allocated memmaps."""
    params = SimulationParams()
    
    # 1. Select System Configuration
    if args and args.system_config:
        params.system_config = args.system_config
    else:
        sys_config = questionary.select(
            "Select System Topology:",
            choices=[
                {"name": "SC-EDT-Target (New proposal: Spacecraft linked to Target via EDT)", "value": "SC_EDT_TARGET"},
                {"name": "Tip-EDT-SC-Rope-Target (Legacy: Tip-EDT-SC with rope to Target)", "value": "SC_ROPE_EDT_TARGET"}
            ]
        ).ask()
        if sys_config is None: sys.exit(0)
        params.system_config = sys_config

    # 2. Select Mission Configuration (Initial State)
    if args and args.mission_config:
        params.mission_config = args.mission_config
    else:
        if params.system_config == 'SC_EDT_TARGET':
            alignment_choices = [
                {"name": "Radial (SC-EDT-Target aligned along local vertical)", "value": "RADIAL"},
                {"name": "Full In-Track (SC-EDT-Target laid along the velocity direction)", "value": "FULL_IN_TRACK"}
            ]
        else:
            alignment_choices = [
                {"name": "Radial (All components radially aligned)", "value": "RADIAL"},
                {"name": "Perpendicular (SC/Target in-track, EDT radial)", "value": "PERPENDICULAR"}
            ]

        config = questionary.select(
            "Select Initial Alignment:",
            choices=alignment_choices
        ).ask()
        if config is None: sys.exit(0)
        params.mission_config = config
    
    # Apply numerical overrides
    if args:
        if args.target_mass is not None:
            params.m_target = args.target_mass
        if args.edt_length is not None:
            params.L_edt = args.edt_length
        if args.inclination is not None:
            params.inc = np.radians(args.inclination)
    
    # 3. Control Setup
    if args is None:
        params.control_enable = questionary.confirm("Enable Closed-Loop Libration Control?").ask()
        if params.control_enable is None: sys.exit(0)
    else:
        params.control_enable = args.control
    
    if params.control_enable:
        limit_deg = questionary.text("Libration Angle Limit [deg]:", default="20.0").ask()
        if limit_deg is None: sys.exit(0)
        params.pitch_limit = np.radians(float(limit_deg))
        
        kp = questionary.text("Control Gain Kp [V/rad]:", default="50.0").ask()
        if kp is None: sys.exit(0)
        params.k_p = float(kp)

    X0 = setup_initial_state(params)
    run_name = f"run_{len(os.listdir('results'))+1:03d}{uuid.uuid4().hex[:4]}"
    run_folder = get_results_folder(run_name, base_dir="results")
    
    n_total = int(t_end * sampling_hz) + 1
    n_state = len(X0)
    
    # Pre-allocate memmaps
    path_t = os.path.join(run_folder, "history_t.dat")
    path_X = os.path.join(run_folder, "history_X.dat")
    hist_t = np.memmap(path_t, dtype='float64', mode='w+', shape=(n_total,))
    hist_X = np.memmap(path_X, dtype='float64', mode='w+', shape=(n_total, n_state))
    
    # Store initial state
    hist_t[0] = 0.0
    hist_X[0] = X0
    
    # Save Initial Config
    dummy_t = np.array([0.0])
    dummy_X = X0.reshape(1, -1)
    dummy_sma = calculate_com_sma(dummy_t, dummy_X, params.to_numba_params(), params)
    save_config_params_results_yaml("config_params_results.yaml", run_folder, dummy_t, dummy_sma, params, params.to_numba_params())
    
    print(f"Starting new simulation: {run_name} ({params.system_config} - {params.mission_config})")
    return run_folder, 0.0, X0, params, hist_t, hist_X, 0, 0.0

def run_mission(args=None):
    skip_checkpoint = args.no_checkpoint if args else False
    # Default to skipping test unless explicitly asked
    skip_test = not args.test if args else True
    method = args.method if args else 'RK45'
    
    sampling_hz = 1.0
    t_end = 24 * 60 * 60 # 10 hours
    step_size = 100000.0

    # 1. Setup Phase
    rf, t_start, X_curr, params, hist_t, hist_X, curr_idx, comp_prev = handle_mission_resumption(t_end, sampling_hz, args=args)
    if rf is None: # New run
        rf, t_start, X_curr, params, hist_t, hist_X, curr_idx, comp_prev = initialize_new_mission(t_end, sampling_hz, args=args)

    p_arr = params.to_numba_params()

    # 2. Stability Guard Phase (Pre-flight)
    if t_start == 0.0 and not skip_test:
        is_stable, msg = run_preflight_stability_check(X_curr, p_arr, params, method)
        if not is_stable:
            print(f"CRITICAL: Simulation aborted during pre-flight. {msg}")
            return

    session_start = time.time()
    print(f"\n--- Starting Simulation ---\nMethod: {method}\nTotal Duration: {t_end/3600:.2f} hours\nCheckpointing: {'Disabled' if skip_checkpoint else 'Enabled'}\n")
    
    t_curr = t_start
    # 3. Execution Phase (Segmented Integration Loop)
    with tqdm(total=int(t_end), initial=int(t_start), unit='s', desc="Mission Progress") as pbar:
        while t_curr < t_end:
            t_next = min(t_curr + step_size, t_end)
            
            # Calculate indices for memmap slicing
            # n_seg = (t_next - t_curr) * sampling_hz
            n_seg = int((t_next - t_curr) * sampling_hz)
            next_idx = curr_idx + n_seg
            
            # Slices (include current point as starting point for integrator)
            # engine.integrate_system will fill these in-place
            sol = integrate_system(X_curr, (t_curr, t_next), p_arr, pbar=pbar, sampling_hz=sampling_hz, method=method)
            
            # Use t_curr to determine if we should include the first point in CSV
            is_first_seg = (t_curr == 0.0 and not os.path.exists(os.path.join(rf, "simulation_results.csv")))

            # Copy solution to memmap
            # Note: sol.t[0] and sol.y[:,0] are the same as t_curr and X_curr
            # We overwrite the segment in memmap starting from curr_idx
            len_sol = len(sol.t)
            hist_t[curr_idx : curr_idx + len_sol] = sol.t
            hist_X[curr_idx : curr_idx + len_sol] = sol.y.T
            
            # Update state for next segment
            t_curr = sol.t[-1]
            X_curr = sol.y[:, -1]
            curr_idx = curr_idx + len_sol - 1
            
            # Binary Checkpoint (Metadata + Incremental CSV)
            if not skip_checkpoint:
                pbar.set_postfix_str("Checkpointing...")
                total_comp = comp_prev + (time.time() - session_start)
                save_checkpoint(rf, t_curr, X_curr, p_arr, current_idx=curr_idx, total_compute_time=total_comp)
                
                # Silent CSV update (append only the new segment)
                if is_first_seg:
                    t_app, y_app = sol.t, sol.y.T
                else:
                    t_app, y_app = sol.t[1:], sol.y.T[1:]
                
                if len(t_app) > 0:
                    tel_seg = post_process_telemetry(t_app, y_app, p_arr, params, include_sma=True)
                    save_csv("simulation_results.csv", rf, t_app, tel_seg, y_app, params, silent=True, append=True)
                pbar.set_postfix_str("")

            # Mid-loop Health Check
            is_sane, reason = check_state_sanity(X_curr, params)
            if not is_sane:
                print(f"\nCRITICAL: Simulation became unstable at t={t_curr:.1f}s. {reason}")
                break

    # 4. Finalization Phase
    total_compute_time = comp_prev + (time.time() - session_start)
    print(f"\n--- Mission Complete ---")
    print(f"Session Compute Time: {(time.time() - session_start)/60:.2f} minutes")
    print(f"Total Compute Time (All Sessions): {total_compute_time/60:.2f} minutes")
    print(f"Total Simulated Time: {t_curr/3600:.2f} hours")

    # Flush memmaps to disk
    hist_t.flush()
    hist_X.flush()

    # Post-process full history from memmap
    final_t = hist_t[:curr_idx+1]
    final_X = hist_X[:curr_idx+1]
    
    sma_final = calculate_com_sma(final_t, final_X, p_arr, params)
    res = calculate_mission_results(final_t, sma_final, params, total_compute_time)
    
    if res:
        print(f"--- Statistical Mission Results ---")
        print(f"Total SMA Drop (Raw): {res['com_sma_drop_total_m']/1000.0:.3f} km")
        print(f"Mean Decay Rate: {res['mean_decay_rate_mps']:.4f} m/s ({res['mean_decay_rate_kmhr']:.4f} km/hr)")
        print(f"Projected Decay: {res['mean_decay_per_orbit_m']:.3f} m/orbit == {res['mean_decay_rate_kmyear']:.2f} km/year")

    # Final Save Phase (Always save full telemetry at the end)
    print("Saving final telemetry to CSV...")
    telemetry_final = post_process_telemetry(final_t, final_X, p_arr, params, include_sma=True, sma_array=sma_final)
    save_csv("simulation_results.csv", rf, final_t, telemetry_final, final_X, params, silent=True)

    save_config_params_results_yaml("config_params_results.yaml", rf, final_t, sma_final, params, p_arr, is_final=True, total_compute_time=total_compute_time)
    plot_simulation(final_t, sma_final, final_X, params, rf)


def parse_arguments():
    """Handles command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-body EDT Simulation with Checkpointing")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Skip periodic binary checkpoints and intermediate CSV saves for maximum performance.")
    parser.add_argument("--test", action="store_true",
                        help="Run validation and pre test for numerical stability.")
    parser.add_argument("--method", choices=['RK45', 'VERLET', 'LSODA', 'RADAU'], default='RK45',
                        help="Integrator: Numba DP5(4) RK45 (default), Velocity Verlet, or numbalsoda LSODA.")
    # Batch run parameters
    parser.add_argument("--target-mass", type=float, help="Target satellite mass [kg]")
    parser.add_argument("--edt-length", type=float, help="EDT length [m]")
    parser.add_argument("--inclination", type=float, help="Inclination [deg]")
    parser.add_argument("--system-config", choices=['SC_EDT_TARGET', 'SC_ROPE_EDT_TARGET'], help="System topology")
    parser.add_argument("--mission-config", help="Initial alignment")
    parser.add_argument("--control", action="store_true", help="Enable closed-loop control")
    parser.add_argument("--resume", action="store_true", help="Prompt to resume an interrupted simulation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_mission(args=args)
