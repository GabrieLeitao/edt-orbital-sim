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

def plot_validation(run_folder, t_vals, energy, rope, pitch, edt, params):
    """Generate structural validation plots"""
    rel_energy_error = (energy - energy[0]) / np.abs(energy[0])
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(t_vals, rel_energy_error); plt.grid(True); plt.title("Energy Conservation Error")
    
    plt.subplot(2, 2, 2)
    plt.plot(t_vals, rope); plt.axhline(y=params.L_rope, color='r', ls='--'); plt.title("Rope Length [m]")
    
    plt.subplot(2, 2, 3)
    plt.plot(t_vals, pitch); plt.grid(True); plt.title("Libration (Pitch) [deg]")
    
    plt.subplot(2, 2, 4)
    plt.plot(t_vals, edt); plt.axhline(y=params.L_edt, color='r', ls='--'); plt.title("EDT Integrity [m]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "validation_plots.png"))
    plt.show()

def run_validation():
    """Main Validation Driver"""
    params = SimulationParams()
    params.I_edt = 0.0 # Conservative
    params.Cd = 0.0
    p_arr = params.to_numba_params()
    
    # 1. Initialize
    X0 = setup_initial_state(params)
    
    t_end = 1000
    method = 'RK45'
    print(f"\n--- Starting Validation ---\nMethod: {method}\nTotal Duration: {t_end/3600:.2f} hours\n")
    
    # 2. Propagate
    with tqdm(total=int(t_end), unit='s', desc="Validating Physics") as pbar:
        sol = integrate_system(X0, (0, t_end), p_arr, rtol=1e-7, atol=1e-9,
                               pbar=pbar, sampling_hz=1.0, method=method)
    
    run_folder = get_results_folder("validation")

    # 3. Analyze & Export
    telemetry = post_process_telemetry(sol.t, sol.y.T, p_arr, params)
    energy = telemetry["energy_j"]
    rope = telemetry["rope_l_m"]
    edt = telemetry["edt_l_m"]
    pitch = telemetry["pitch_deg"]

    rel_energy_error = (energy - energy[0]) / np.abs(energy[0])
    telemetry["rel_energy_error"] = rel_energy_error

    save_csv("validation_results.csv", run_folder, sol.t, telemetry, sol.y.T, params)
    
    # 4. Visualize
    plot_validation(run_folder, sol.t, energy, rope, pitch, edt, params)
    
    print(f"\n--- Validation Report ---\n1. Energy Stability: {np.max(np.abs(rel_energy_error)):.2e}")
    print(f"2. Structure Check: {'Stable' if np.max(rope) < params.L_rope * 1.1 else 'UNSTABLE'}")

if __name__ == "__main__":
    run_validation()
