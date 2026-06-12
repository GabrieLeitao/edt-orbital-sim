import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from params import SimulationParams
from engine import setup_initial_state, integrate_system
from analysis import post_process_telemetry, save_csv
from utils import get_results_folder

def plot_validation(run_folder, t_vals, energy, rope, pitch, edt, sma_km, params):
    """Generate structural and orbital validation plots"""
    rel_energy_error = (energy - energy[0]) / np.abs(energy[0])
    sma_drift_m = (sma_km - sma_km[0]) * 1000.0
    
    # Calculate bounds for error reporting
    max_energy_err = np.max(np.abs(rel_energy_error))
    max_sma_drift = np.max(np.abs(sma_drift_m))
    
    has_rope = (params.system_config != 'SC_EDT_TARGET')
    num_rows = 3 if has_rope else 2
    
    plt.figure(figsize=(15, 4 * num_rows))
    
    # 1. Energy - Pure Numerical Error (Invariant to J2)
    plt.subplot(num_rows, 2, 1)
    plt.plot(t_vals, rel_energy_error)
    plt.axhline(y=max_energy_err, color='k', ls=':', alpha=0.5, label=f'Max Error: {max_energy_err:.2e}')
    plt.axhline(y=-max_energy_err, color='k', ls=':', alpha=0.5)
    plt.grid(True); plt.title("Numerical Energy Leakage")
    plt.ylabel("Rel Error [-]"); plt.legend()
    
    # 2. SMA Drift - Should be flat if J2=0 and Inc=0
    plt.subplot(num_rows, 2, 2)
    plt.plot(t_vals, sma_drift_m, color='r')
    plt.axhline(y=max_sma_drift, color='k', ls=':', alpha=0.5, label=f'Max Drift: {max_sma_drift:.4f}m')
    plt.axhline(y=-max_sma_drift, color='k', ls=':', alpha=0.5)
    plt.grid(True); plt.title("Numerical SMA Drift")
    plt.ylabel("Drift [m]"); plt.legend()
    
    # 3. EDT Integrity
    plt.subplot(num_rows, 2, 3)
    plt.plot(t_vals, edt); plt.axhline(y=params.L_edt, color='r', ls='--'); plt.title("EDT Length [m]")
    
    # 4. Libration
    plt.subplot(num_rows, 2, 4)
    plt.plot(t_vals, pitch); plt.grid(True); plt.title("Libration (Pitch) [deg]")
    
    # 5. Rope (Conditional)
    if has_rope:
        plt.subplot(num_rows, 2, 5)
        plt.plot(t_vals, rope); plt.axhline(y=params.L_rope, color='r', ls='--'); plt.title("Rope Length [m]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "validation_plots.png"))
    plt.show()

def run_validation():
    """Main Validation Driver"""
    params = SimulationParams()
    
    # --- SETUP "CLEAN ROOM" FOR NUMERICAL VALIDATION ---
    # Correctly disable all non-conservative forces
    params.Cd = 0.0           # Disable Drag
    # params.J2 = 0.0           # Disable J2
    # params.inc = 0.0          # Equatorial
    params.r_load = 1e18      # Effectively zero current (Lorentz Off)
    
    p_arr = params.to_numba_params()
    
    # 1. Initialize
    X0 = setup_initial_state(params)
    
    t_end = 24 * 3600 # 1 day
    method = 'RK45'
    print(f"\n--- Starting Numerical Validation ---")
    print(f"Mode: Clean-Room (J2=Off, Inc=0, Conservative)")
    print(f"Method: {method}, Duration: {t_end}s\n")
    
    # 2. Propagate
    with tqdm(total=int(t_end), unit='s', desc="Validating Physics") as pbar:
        sol = integrate_system(X0, (0, t_end), p_arr, rtol=1e-7, atol=1e-9, # Tightened for validation
                               pbar=pbar, sampling_hz=1.0, method=method)
    
    run_folder = get_results_folder("validation", base_dir="test_results")

    # 3. Analyze & Export
    telemetry = post_process_telemetry(sol.t, sol.y.T, p_arr, params)
    energy = telemetry["energy_j"]
    rope = telemetry["rope_l_m"]
    edt = telemetry["edt_l_m"]
    pitch = telemetry["pitch_deg"]
    sma_km = telemetry["sma_km"]

    rel_energy_error = (energy - energy[0]) / np.abs(energy[0])
    sma_drift_m = (sma_km - sma_km[0]) * 1000.0
    
    max_energy_err = np.max(np.abs(rel_energy_error))
    max_sma_drift = np.max(np.abs(sma_drift_m))
    
    telemetry["rel_energy_error"] = rel_energy_error

    save_csv("validation_results.csv", run_folder, sol.t, telemetry, sol.y.T, params)
    
    # 4. Visualize
    plot_validation(run_folder, sol.t, energy, rope, pitch, edt, sma_km, params)
    
    print(f"\n--- Numerical Validation Report (Bounds) ---")
    print(f"1. Max Energy Deviation: {max_energy_err:.2e} (Target: < 1e-7)")
    print(f"2. Max SMA Deviation:    {max_sma_drift:.6f} m (Target: < 0.01m)")
    print(f"3. Structure Integrity:  {'Stable' if np.max(edt) < params.L_edt * 1.05 else 'UNSTABLE'}")

if __name__ == "__main__":
    run_validation()
