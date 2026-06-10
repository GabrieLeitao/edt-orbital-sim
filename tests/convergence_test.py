import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from params import SimulationParams
from engine import setup_initial_state, integrate_system
from analysis import calculate_com_sma, calculate_total_energy_fast, get_mass_fast
from utils import get_results_folder

def run_single_n(n, t_end, method):
    """
    Worker function to run a single simulation for a given N_edt.
    """
    try:
        params = SimulationParams()
        params.N_edt = n
        
        # --- CLEAN ROOM ISOLATION (Consistent with validate_physics) ---
        params.Cd = 0.0            # Conservative
        params.J2 = 0.0            # Disable J2
        params.inc = 0.0           # Disable inclination
        params.r_load = 1e18       # Disable Lorentz
        
        p_arr = params.to_numba_params()
        
        X0 = setup_initial_state(params)
        num_masses = params.num_masses
        masses = np.array([get_mass_fast(i, p_arr, num_masses) for i in range(num_masses)])
        
        start_time = time.time()
        # Use tighter tolerances for convergence to isolate spatial error from solver error
        sol = integrate_system(X0, (0, t_end), p_arr, rtol=1e-7, atol=1e-9,
                               pbar=None, method=method)
        end_time = time.time()
        
        comp_time = end_time - start_time
        
        # Calculate SMA
        sma = calculate_com_sma(sol.t, sol.y.T, p_arr, params)
        sma_drift = np.abs(sma[-1] - sma[0])
        
        # Calculate Energy Error
        p0 = X0[:3*num_masses].reshape((num_masses, 3))
        v0 = X0[3*num_masses:].reshape((num_masses, 3))
        e0 = calculate_total_energy_fast(p0, v0, masses, p_arr)
        
        pf = sol.y[:3*num_masses, -1].reshape((num_masses, 3))
        vf = sol.y[3*num_masses:, -1].reshape((num_masses, 3))
        ef = calculate_total_energy_fast(pf, vf, masses, p_arr)
        
        rel_energy_error = np.abs((ef - e0) / e0)
        
        return {
            "n": n,
            "time": comp_time,
            "drift": sma_drift,
            "energy_error": rel_energy_error,
            "final_sma": sma[-1],
            "status": "success"
        }
    except Exception as e:
        return {
            "n": n,
            "status": "error",
            "error": str(e)
        }

def run_convergence():
    """Performs spatial convergence test by varying N_edt using parallel execution."""
    n_edt_values = [2, 4, 12, 20]
    results = []
    
    t_end = 4 * 3600 
    method = 'VERLET'
    
    print(f"--- Starting ISOLATED Spatial Convergence Test ---")
    print(f"Mode: Clean-Room (J2=Off, Inc=0, Conservative)")
    print(f"Method: {method}, Duration: {t_end}s")
    
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"Workers: {min(len(n_edt_values), num_workers)}\n")

    run_folder = get_results_folder("convergence", base_dir="test_results")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_n, n, t_end, method): n for n in n_edt_values}
        
        with tqdm(total=len(n_edt_values), desc="Convergence") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res["status"] == "success":
                    results.append(res)
                else:
                    tqdm.write(f"Error for N_edt={res['n']}: {res.get('error')}")
                pbar.update(1)

    if not results: return

    results.sort(key=lambda x: x['n'])
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(run_folder, f"convergence_results_{method}.csv")
    df[["n", "time", "drift", "energy_error", "final_sma"]].to_csv(csv_path, index=False)
    print(f"\nCSV results saved to {csv_path}")

    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(df["n"], df["time"], 'o-')
    plt.xlabel("N_edt (Segments)"); plt.ylabel("Time (s)"); plt.title("Compute Cost"); plt.grid(True)
    # 2. SMA Drift
    plt.subplot(1, 3, 2)
    # Add a tiny epsilon to prevent log(0) warnings
    plt.plot(df["n"], df["drift"] + 1e-18, 's-', color='r')
    plt.xlabel("N_edt (Segments)"); plt.ylabel("SMA Drift (m)"); plt.yscale('log'); plt.title("Numerical Stability"); plt.grid(True)

    # 3. Energy Error
    plt.subplot(1, 3, 3)
    # Add a tiny epsilon to prevent log(0) warnings
    plt.plot(df["n"], df["energy_error"] + 1e-18, 'd-', color='g')
    plt.xlabel("N_edt (Segments)"); plt.ylabel("Rel Energy Error [-]"); plt.yscale('log'); plt.title("Numerical Integrity"); plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(run_folder, f"convergence_results_{method}.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    print("\nConvergence test complete.")

if __name__ == "__main__":
    run_convergence()
