import numpy as np
import time
import matplotlib.pyplot as plt
from params import SimulationParams
from engine import setup_initial_state, integrate_system
from analysis import calculate_com_sma

def run_convergence():
    """Performs spatial convergence test by varying N_edt."""
    n_edt_values = [2, 5, 10, 15, 20]
    results = []
    
    t_end = 3600 # 1 hour
    method = 'RK45'
    
    print(f"--- Starting Spatial Convergence Test ---")
    print(f"Method: {method}, Duration: {t_end}s\n")
    
    for n in n_edt_values:
        params = SimulationParams()
        params.N_edt = n
        # Disable external forces for a clean numerical test
        params.I_edt = 0.0
        params.Cd = 0.0
        
        p_arr = params.to_numba_params()
        X0 = setup_initial_state(params)
        
        start_time = time.time()
        sol = integrate_system(X0, (0, t_end), p_arr, f"N={n}", rtol=1e-7, atol=1e-9, method=method)
        end_time = time.time()
        
        comp_time = end_time - start_time
        sma = calculate_com_sma(sol.t, sol.y.T, p_arr, params)
        sma_drift = np.abs(sma[-1] - sma[0])
        
        results.append({
            "n": n,
            "time": comp_time,
            "drift": sma_drift,
            "final_sma": sma[-1]
        })
        
        print(f"N_edt={n:2d} | Compute: {comp_time:6.2f}s | SMA Drift: {sma_drift:8.2f}m")

    # Visualization
    n_vals = [r["n"] for r in results]
    t_vals = [r["time"] for r in results]
    d_vals = [r["drift"] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_vals, t_vals, 'o-')
    plt.xlabel("N_edt (Segments)")
    plt.ylabel("Compute Time (s)")
    plt.title("Execution Scalability")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(n_vals, d_vals, 's-', color='r')
    plt.xlabel("N_edt (Segments)")
    plt.ylabel("SMA Drift (m)")
    plt.yscale('log')
    plt.title("Numerical Convergence (SMA Drift)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("convergence_results.png")
    print(f"\nResults saved to convergence_results.png")
    plt.show()

if __name__ == "__main__":
    run_convergence()
