import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import questionary
import hashlib
import time
import argparse
from tqdm import tqdm

from params import SimulationParams
from engine import setup_initial_state, integrate_system, save_checkpoint, load_checkpoint
from analysis import calculate_com_sma, save_csv, save_config_params_results_yaml, post_process_telemetry
from utils import get_results_folder

from stability import run_preflight_stability_check, check_state_sanity

def plot_simulation(t_vals, sma_com, X_vals, params, run_folder):
    """Generate deorbiting plots"""
    plt.figure(figsize=(14, 6))
    
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
    plt.subplot(1, 2, 2)
    final_pos = X_vals[-1, :3*params.num_masses].reshape((params.num_masses, 3))
    rel_pos = final_pos - final_pos[params.N_edt + 1]
    plt.plot(rel_pos[:, 0], rel_pos[:, 1], '-ok')
    plt.gca().set_aspect('equal'); plt.grid(True)
    plt.title('Final Tether Configuration')
    plt.xlabel('Radial Distance [m]')
    plt.ylabel('In-Track Distance [m]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "simulation_plots.png"))
    plt.show()

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

def handle_mission_resumption(t_end, sampling_hz):
    """Manages the logic for resuming an interrupted simulation using memmaps."""
    resumable_runs = find_resumable_runs()
    if not resumable_runs:
        return None, 0.0, None, None, None, None, 0, 0.0

    use_checkpoint = questionary.confirm("Found resumable runs. Would you like to resume?").ask()
    if not use_checkpoint:
        return None, 0.0, None, None, None, None, 0, 0.0

    run_name = questionary.select("Select run to resume:", choices=resumable_runs).ask()
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

def initialize_new_mission(t_end, sampling_hz):
    """Sets up a fresh simulation run with pre-allocated memmaps."""
    params = SimulationParams()
    X0 = setup_initial_state(params)
    run_name = f"run_{len(os.listdir('results'))+1:03d}"
    run_folder = get_results_folder(run_name)
    
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
    
    print(f"Starting new simulation: {run_name}")
    return run_folder, 0.0, X0, params, hist_t, hist_X, 0, 0.0

def run_mission(skip_checkpoint=False, skip_test=False, method='RK45'):
    # 0. Constants
    sampling_hz = 1.0
    t_end = 5400 * 2 # 2 orbits
    step_size = 1000.0

    # 1. Setup Phase
    rf, t_start, X_curr, params, hist_t, hist_X, curr_idx, comp_prev = handle_mission_resumption(t_end, sampling_hz)
    if rf is None: # New run
        rf, t_start, X_curr, params, hist_t, hist_X, curr_idx, comp_prev = initialize_new_mission(t_end, sampling_hz)

    p_arr = params.to_numba_params()

    # 2. Stability Guard Phase (Pre-flight)
    if t_start == 0.0 and not skip_test:
        is_stable, msg = run_preflight_stability_check(X_curr, p_arr, params, method)
        if not is_stable:
            print(f"CRITICAL: Simulation aborted during pre-flight. {msg}")
            return

    session_start = time.time()
    print(f"\n--- Starting Simulation ---\nMethod: {method}\nTotal Duration: {t_end/3600:.2f} hours\nCheckpointing: {'Disabled' if skip_checkpoint else 'Enabled'}\nPre-flight Test: {'Skipped' if skip_test else 'Enabled'}\n")
    
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
            
            # Mid-loop Health Check
            is_sane, reason = check_state_sanity(X_curr, params)
            if not is_sane:
                print(f"\nCRITICAL: Simulation became unstable at t={t_curr:.1f}s. {reason}")
                break
            
            # Binary Checkpoint (Metadata only)
            if not skip_checkpoint:
                pbar.set_postfix_str("Checkpointing...")
                total_comp = comp_prev + (time.time() - session_start)
                save_checkpoint(rf, t_curr, X_curr, p_arr, current_idx=curr_idx, total_compute_time=total_comp)
                
                # Silent CSV update (append only the new segment)
                # Performance: include_sma=False avoids expensive orbital calculations mid-loop
                tel_seg = post_process_telemetry(sol.t, sol.y.T, p_arr, params, include_sma=False)
                save_csv("simulation_results.csv", rf, sol.t, tel_seg, sol.y.T, params, silent=True, append=True)
                pbar.set_postfix_str("")

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
    save_config_params_results_yaml("config_params_results.yaml", rf, final_t, sma_final, params, p_arr, is_final=True, total_compute_time=total_compute_time)
    plot_simulation(final_t, sma_final, final_X, params, rf)

def parse_arguments():
    """Handles command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-body EDT Simulation with Checkpointing")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Skip periodic binary checkpoints and intermediate CSV saves for maximum performance.")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip validation and pre test for numerical stability.")
    parser.add_argument("--method", choices=['RK45', 'VERLET', 'LSODA', 'RADAU'], default='RK45',
                        help="Integrator: Numba DP5(4) RK45 (default), Velocity Verlet, or numbalsoda LSODA.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # cProfile.run("run_mission(skip_checkpoint=args.no_checkpoint, skip_test=args.no_test, method=args.method)", "profile_results.prof")
    run_mission(skip_checkpoint=args.no_checkpoint, skip_test=args.no_test, method=args.method)