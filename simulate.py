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
from analysis import calculate_com_sma, save_csv, save_config_params_results_yaml
from utils import get_results_folder

def plot_simulation(t_vals, sma_com, X_vals, params, run_folder):
    """Generate deorbiting plots"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_vals/60, (sma_com - sma_com[0])/1e3)
    plt.grid(True); plt.xlabel('Time [min]'); plt.ylabel('Δ System SMA (CoM) [km]')
    plt.title('System Orbital Decay')
    
    plt.subplot(1, 2, 2)
    final_pos = X_vals[-1, :3*params.num_masses].reshape((params.num_masses, 3))
    rel_pos = final_pos - final_pos[params.N_edt + 2]
    plt.plot(rel_pos[:, 0], rel_pos[:, 1], '-ok')
    plt.gca().set_aspect('equal'); plt.grid(True); plt.title('Final Tether Configuration')
    
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
        return [], []
        
    df = pd.read_csv(csv_path)
    all_t = df['time_s'].values.tolist()
    interleaved = df.iloc[:, 2:].values
    n_m = params.num_masses
    pos_rec = np.zeros((len(all_t), 3*n_m))
    vel_rec = np.zeros((len(all_t), 3*n_m))
    for i in range(n_m):
        pos_rec[:, 3*i:3*i+3] = interleaved[:, 6*i:6*i+3]
        vel_rec[:, 3*i:3*i+3] = interleaved[:, 6*i+3:6*i+6]
    all_X = np.hstack([pos_rec, vel_rec]).tolist()
    return all_t, all_X

def handle_mission_resumption():
    """Manages the logic for resuming an interrupted simulation."""
    resumable_runs = find_resumable_runs()
    if not resumable_runs:
        return None, 0.0, None, None, [], []

    use_checkpoint = questionary.confirm("Found resumable runs. Would you like to resume?").ask()
    if not use_checkpoint:
        return None, 0.0, None, None, [], []

    run_name = questionary.select("Select run to resume:", choices=resumable_runs).ask()
    run_folder = os.path.join('results', run_name)
    
    yaml_path = os.path.join(run_folder, "config_params_results.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: YAML configuration missing for {run_name}. Cannot resume safely.")
        return None, 0.0, None, None, [], []

    params = SimulationParams.from_yaml(yaml_path)
    p_hash_curr = hashlib.sha256(params.to_numba_params().tobytes()).hexdigest()
    
    t_s, X0, p_flat, p_hash_stored, h_t, h_X = load_checkpoint(run_folder)
    
    if p_hash_stored and p_hash_curr != p_hash_stored:
        print(f"CRITICAL ERROR: Parameter mismatch detected! Aborting.")
        return None, 0.0, None, None, [], []

    # History recovery (Binary primary, CSV fallback)
    if h_t is not None and len(h_t) > 0:
        all_t, all_X = h_t.tolist(), h_X.tolist()
    else:
        all_t, all_X = recover_history_from_csv(run_folder, params)

    if all_t and np.isclose(all_t[-1], t_s):
        all_t.pop(); all_X.pop()

    print(f"Resuming mission from t = {t_s:.1f}s")
    return run_folder, t_s, X0, params, all_t, all_X

def initialize_new_mission():
    """Sets up a fresh simulation run."""
    params = SimulationParams()
    X0 = setup_initial_state(params)
    run_name = f"run_{len(os.listdir('results'))+1:03d}"
    run_folder = get_results_folder(run_name)
    
    # Save Initial Config
    dummy_t = np.array([0.0])
    dummy_X = X0.reshape(1, -1)
    dummy_sma = calculate_com_sma(dummy_t, dummy_X, params.to_numba_params(), params)
    save_config_params_results_yaml("config_params_results.yaml", run_folder, dummy_t, dummy_sma, params, params.to_numba_params())
    
    print(f"Starting new simulation: {run_name}")
    return run_folder, 0.0, X0, params, [], []

def run_mission(skip_checkpoint=False):
    """Main Orchestrator following KSRP principles."""
    # 1. Setup Phase
    rf, t_start, X0, params, all_t, all_X = handle_mission_resumption()
    if rf is None: # New run
        rf, t_start, X0, params, all_t, all_X = initialize_new_mission()

    p_arr = params.to_numba_params()
    t_end = 700
    step_size = 10.0
    t_curr, X_curr = t_start, X0
    real_start = time.time()
    
    # 2. Execution Phase (Segmented Integration Loop)
    with tqdm(total=int(t_end), initial=int(t_start), unit='s', desc="Mission Progress") as pbar:
        while t_curr < t_end:
            t_next = min(t_curr + step_size, t_end)
            
            # Performance: Sampling at 1Hz prevents memory bloat (checkpoint stays < 10MB)
            sol = integrate_system(X_curr, (t_curr, t_next), p_arr, desc="", pbar=pbar, sampling_hz=1.0)
            
            # Segment Data
            seg_t = sol.t
            seg_X = sol.y.T
            
            all_t.extend(seg_t); all_X.extend(seg_X)
            t_curr, X_curr = seg_t[-1], seg_X[-1]
            
            # Binary Checkpoint + Silent CSV (if enabled)
            if not skip_checkpoint:
                pbar.set_postfix_str("Checkpointing...")
                
                # Checkpoint keeps full history for lossless resume
                save_checkpoint(rf, t_curr, X_curr, p_arr, np.array(all_t), np.array(all_X))
                
                # Performance: Append only the NEW segment to CSV to avoid RAM spikes
                sma_seg = calculate_com_sma(np.array(seg_t), np.array(seg_X), p_arr, params)
                save_csv("simulation_results.csv", rf, np.array(seg_t), sma_seg, "sma_com_km", np.array(seg_X), params, silent=True, append=True)
                
                pbar.set_postfix_str("")

    # 3. Finalization Phase
    print(f"\n--- Mission Complete ---\nTotal Compute Time: {(time.time() - real_start)/60:.2f} minutes")
    all_t, all_X = np.array(all_t), np.array(all_X)
    sma_final = calculate_com_sma(all_t, all_X, p_arr, params)
    save_config_params_results_yaml("config_params_results.yaml", rf, all_t, sma_final, params, p_arr, is_final=True)
    plot_simulation(all_t, sma_final, all_X, params, rf)

def parse_arguments():
    """Handles command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-body EDT Simulation with Checkpointing")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Skip periodic binary checkpoints and intermediate CSV saves for maximum performance.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_mission(skip_checkpoint=args.no_checkpoint)
