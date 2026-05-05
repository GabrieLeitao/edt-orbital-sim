import matplotlib.pyplot as plt
import os
import numpy as np

from params import SimulationParams
from engine import setup_initial_state, integrate_system
from analysis import calculate_com_sma, save_csv

def plot_simulation(t_vals, sma_com, X_vals, params):
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
    plt.savefig(os.path.join("results", "simulation_plots.png"))
    plt.show()

def run_mission():
    """Main Mission Driver"""
    params = SimulationParams()
    p_arr = params.to_numba_params()
    
    # 1. Initialize
    X0 = setup_initial_state(params)
    
    # 2. Propagate
    sol = integrate_system(X0, (0, 5400), p_arr, "Propagating Deorbit")
    
    # 3. Analyze & Export
    sma_com = calculate_com_sma(sol.t, sol.y.T, p_arr, params)
    save_csv("simulation_results.csv", sol.t, sma_com, "sma_com_km", sol.y.T, params)
    
    # 4. Visualize
    plot_simulation(sol.t, sma_com, sol.y.T, params)

if __name__ == "__main__":
    run_mission()
