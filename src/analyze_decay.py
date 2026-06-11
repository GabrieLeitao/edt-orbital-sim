import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def load_simulation_data(results_dir="results"):
    """
    Scans the results directory for YAML files and extracts key metrics.
    """
    data = []
    yaml_pattern = os.path.join(results_dir, "run_*", "config_params_results.yaml")
    paths = glob.glob(yaml_pattern)

    files = [
        p for p in paths
        if int(os.path.basename(os.path.dirname(p)).split("_")[1][:3]) > 44
    ]
    
    if not files:
        print(f"No result files found in {results_dir}")
        return pd.DataFrame()

    for f in files:
        try:
            with open(f, 'r') as stream:
                content = yaml.safe_load(stream)
                
                # Extract parameters
                params = content.get('parameters', {})
                masses = params.get('system_masses', {})
                edt = params.get('edt_properties', {})
                iorbit = params.get('initial_orbit', {})
                results = content.get('results', {})
                
                m_target = masses.get('m_target')
                m_sc = masses.get('m_sc', 100.0)
                l_edt = edt.get('L_edt')
                inc_rad = iorbit.get('inc_rad')
                alt = iorbit.get('alt')
                decay_rate = results.get('mean_decay_rate_kmyear')
                
                if l_edt != 2500 and m_target is not None and l_edt is not None and decay_rate is not None and inc_rad is not None and alt is not None:
                    inc_deg = np.degrees(inc_rad) if inc_rad is not None else 0.0
                    data.append({
                        'run_id': content.get('metadata', {}).get('run_id'),
                        'm_target': m_target,
                        'm_total': m_target + m_sc,
                        'l_edt': l_edt,
                        'inclination': inc_deg,
                        'altitude_km': alt / 1000.0,
                        'decay_rate_km_yr': decay_rate
                    })
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    df = pd.DataFrame(data).sort_values(by=['inclination', 'm_target', 'l_edt'])
    
    # Filter Outliers: Remove physically unrealistic data points
    if not df.empty:
        initial_count = len(df)
        df = df[(df['decay_rate_km_yr'] > 0) & (df['decay_rate_km_yr'] < 40000)]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} outlier/invalid data points.")
            
    return df


def decay_law_model(vars, c_drag, k_lor):
    """
    Physical Model: Decay = Drag_Term + Lorentz_Term
    D = c_drag + k_lor * (L^2 / M_total)
    """
    l, m_total = vars
    return c_drag + k_lor * (l**2 / m_total)


def generate_plots(df_subset, inc_value, alt_value, c_drag, k_lor):
    """
    Generates and saves the two performance charts for a specific inclination and altitude.
    """
    inc_str = f"{inc_value:.1f}".replace('.', '_')
    alt_str = f"{alt_value:.0f}"
    
    # Pre-calculate smooth ranges for continuous model curves
    l_smooth = np.linspace(df_subset['l_edt'].min() * 0.9 if df_subset['l_edt'].min() > 0 else 0, 
                           df_subset['l_edt'].max() * 1.1, 100)
    m_target_smooth = np.linspace(df_subset['m_target'].min() * 0.9 if df_subset['m_target'].min() > 0 else 0, 
                                  df_subset['m_target'].max() * 1.1, 100)
    m_total_smooth = m_target_smooth + 100.0

    # -----------------------------------------------------------------
    # PROFILE 1: Decay vs. Tether Length
    # -----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    unique_masses = sorted(df_subset['m_target'].unique(), reverse=True)
    colors1 = plt.cm.viridis(np.linspace(0, 0.8, len(unique_masses)))

    for idx, m in enumerate(unique_masses):
        m_df = df_subset[df_subset['m_target'] == m]
        m_total = m + 100.0
        
        ax1.scatter(m_df['l_edt'], m_df['decay_rate_km_yr'], 
                    color=colors1[idx], label=f'Target: {m:.0f}kg', zorder=3, s=45)
        
        curve1 = decay_law_model((l_smooth, np.full_like(l_smooth, m_total)), c_drag, k_lor)
        ax1.plot(l_smooth, curve1, color=colors1[idx], alpha=0.6, linestyle='--')

    ax1.set_title(f"EDT Performance: Decay vs. Tether Length (Inc: {inc_value:.1f}°, Alt: {alt_value:.0f}km)")
    ax1.set_xlabel("EDT Length (m)")
    ax1.set_ylabel("Mean Decay Rate (km/year)")
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left')
    plt.savefig(f"performance_results/decay_vs_length_inc_{inc_str}_alt_{alt_str}.png", bbox_inches='tight')
    plt.close(fig1)

    # -----------------------------------------------------------------
    # PROFILE 2: Decay vs. Target Mass
    # -----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    unique_lengths = sorted(df_subset['l_edt'].unique())
    colors2 = plt.cm.plasma(np.linspace(0, 0.8, len(unique_lengths)))

    for idx, l_val in enumerate(unique_lengths):
        l_df = df_subset[df_subset['l_edt'] == l_val]
        
        ax2.scatter(l_df['m_target'], l_df['decay_rate_km_yr'], 
                    color=colors2[idx], label=f'EDT Length: {l_val:.0f}m', zorder=3, s=45)
        
        curve2 = decay_law_model((np.full_like(m_total_smooth, l_val), m_total_smooth), c_drag, k_lor)
        ax2.plot(m_target_smooth, curve2, color=colors2[idx], alpha=0.6, linestyle='--')

    ax2.set_title(f"EDT Performance: Decay vs. Target Mass (Inc: {inc_value:.1f}°, Alt: {alt_value:.0f}km)")
    ax2.set_xlabel("Target Mass (kg)")
    ax2.set_ylabel("Mean Decay Rate (km/year)")
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')
    plt.savefig(f"performance_results/decay_vs_target_mass_inc_{inc_str}_alt_{alt_str}.png", bbox_inches='tight')
    plt.close(fig2)


def plot_time_to_deorbit(fit_results):
    """
    Estimates and plots the time required to deorbit from 800km to 350km.
    """
    # fit_results: list of (inc, alt, c_drag, k_lor)
    if not fit_results:
        return

    unique_incs = sorted(list(set(r[0] for r in fit_results)))

    h_initial = 800 # km
    h_final = 300 # km
    
    # Range of target masses and tether lengths to consider for estimation
    target_masses = np.linspace(100, 1000, 20)
    l_edt = 1500.0  # UPDATED: Assumed length for estimation

    for inc in unique_incs:
        inc_results = [r for r in fit_results if r[0] == inc]
        if len(inc_results) < 2:
            continue
            
        inc_results.sort(key=lambda x: x[1]) # Sort by altitude
        alts = np.array([r[1] for r in inc_results])
        c_drags = np.array([r[2] for r in inc_results])
        k_lors = np.array([r[3] for r in inc_results])
        
        # Interpolate coefficients
        f_c_drag = interp1d(alts, c_drags, kind='linear', fill_value="extrapolate")
        f_k_lor = interp1d(alts, k_lors, kind='linear', fill_value="extrapolate")
        
        # Integration
        h = np.linspace(h_final, h_initial, 100)
        dh = h[1] - h[0]
        
        times = []
        for m_t in target_masses:
            m_total = m_t + 100.0
            
            # Decay rate at each altitude
            decay_rates = f_c_drag(h) + f_k_lor(h) * (l_edt**2 / m_total)
            
            # Time segment = dh / Rate
            # Ensure no division by zero
            decay_rates = np.maximum(decay_rates, 1e-6)
            
            time_total = np.sum(dh / decay_rates)
            
            # Apply safety cap
            if time_total > 500:
                print(f"  ⚠️ Warning: High deorbit time ({time_total:.1f} years) for Inc: {inc:.1f}°, capping at 500.")
                time_total = 500
            
            times.append(time_total)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(target_masses, times)
        ax.set_title(f"Est. Deorbit Time ({h_initial}km to {h_final}km, L={l_edt}m) - Inc: {inc:.1f}°")
        ax.set_xlabel("Target Mass (kg)")
        ax.set_ylabel("Time to Deorbit (years)")
        ax.grid(True)
        plt.savefig(f"performance_results/est_deorbit_time_inc_{inc:.1f}_alt_{h_initial}_{h_final}".replace('.', '_')+".png", bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved Time-to-Deorbit estimation plot for Inc: {inc:.1f}°")



def plot_time_to_deorbit_vs_inclination(fit_results, target_mass=500.0, l_edt=1500.0):
    """
    Plots the estimated time to deorbit (800km to 350km) vs inclination.
    """
    if not fit_results:
        return

    # Extract unique inclinations and their respective fit results
    unique_incs = sorted(list(set(r[0] for r in fit_results)))
    unique_alts = sorted(list(set(r[1] for r in fit_results)))
    
    # We need to interpolate across altitudes to get deorbit time for each inclination
    # Assuming we have data for different altitudes at each inclination.
    h_initial = 800 # km
    h_final = 300 # km
    
    deorbit_times = []
    
    for inc in unique_incs:
        # Get fit results for this inclination, sorted by altitude
        inc_results = sorted([r for r in fit_results if r[0] == inc], key=lambda x: x[1])
        
        if len(inc_results) < 2:
            continue
            
        alts = np.array([r[1] for r in inc_results])
        c_drags = np.array([r[2] for r in inc_results])
        k_lors = np.array([r[3] for r in inc_results])
        
        # Interpolate coefficients
        f_c_drag = interp1d(alts, c_drags, kind='linear', fill_value="extrapolate")
        f_k_lor = interp1d(alts, k_lors, kind='linear', fill_value="extrapolate")
        
        # Integration from 800 down to 350 km
        h = np.linspace(h_final, h_initial, 100)
        dh = h[1] - h[0]
        
        m_total = target_mass + 100.0
        
        # Decay rate at each altitude
        decay_rates = f_c_drag(h) + f_k_lor(h) * (l_edt**2 / m_total)
        
        # Ensure no division by zero
        decay_rates = np.maximum(decay_rates, 1e-6)
        
        time_total = np.sum(dh / decay_rates)
        
        # Apply safety cap
        if time_total > 500:
            print(f"  ⚠️ Warning: High deorbit time ({time_total:.1f} years) for Inc: {inc:.1f}°, capping at 500.")
            time_total = 500

        deorbit_times.append(time_total)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(unique_incs, deorbit_times, marker='o')
    ax.set_title(f"Deorbit Time vs Inclination ({h_initial}km to {h_final}km)\nTarget Mass={target_mass}kg, EDT Length={l_edt}m")
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Time to Deorbit (years)")
    ax.grid(True)
    plt.savefig(f"performance_results/deorbit_time_vs_inclination_m{target_mass}_L{l_edt}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Deorbit Time vs Inclination estimation plot.")


def analyze_and_plot(df):
    if df.empty:
        return

    print("\n--- Simulation Summary ---")
    print(df.to_string(index=False))

    os.makedirs("performance_results", exist_ok=True)
    unique_incs = sorted(df['inclination'].unique())
    unique_alts = sorted(df['altitude_km'].unique())

    print(f"\n--- Fitting and Generating Plots Separately for Each Inclination and Altitude ---")

    fit_results = []
    for inc in unique_incs:
        for alt in unique_alts:
            subset_df = df[(df['inclination'] == inc) & (df['altitude_km'] == alt)]
            
            if subset_df.empty:
                continue

            print(f"\n=========================================")
            print(f"FIT METRICS FOR INCLINATION: {inc:.1f}°, ALTITUDE: {alt:.0f}km")
            print(f"=========================================")

            if len(subset_df) < 3:
                print(f"  ❌ Not enough data points to fit. Skipping.")
                continue

            try:
                # Perform mathematical curve fitting on the localized 2D surface
                popt, pcov = curve_fit(
                    lambda x, c, k: decay_law_model((x[:, 0], x[:, 1]), c, k),
                    subset_df[['l_edt', 'm_total']].values,
                    subset_df['decay_rate_km_yr'].values
                )
                c_drag, k_lor = popt
                
                # Extract standard errors from covariance diagonals
                perr = np.sqrt(np.diag(pcov))
                c_drag_err, k_lor_err = perr
                
                # Goodness-of-fit metrics calculation
                y_true = subset_df['decay_rate_km_yr'].values
                y_pred = decay_law_model((subset_df['l_edt'].values, subset_df['m_total'].values), c_drag, k_lor)
                
                residuals = y_true - y_pred
                rmse = np.sqrt(np.mean(residuals**2))
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                print(f"  Formula: Decay = c_drag + k_lor * (L^2 / M_total)")
                print(f"  c_drag = {c_drag:.4f} ± {c_drag_err:.4f}")
                print(f"  k_lor  = {k_lor:.6f} ± {k_lor_err:.6f}")
                print(f"  R² Score: {r_squared:.4f}")
                print(f"  RMSE:     {rmse:.2f} km/year")

                fit_results.append((inc, alt, c_drag, k_lor))

            except Exception as e:
                print(f"  Could not fit model: {e}")
                c_drag, k_lor = 50.0, 0.05

            # Call the updated rendering function
            generate_plots(subset_df, inc, alt, c_drag, k_lor)
            print(f"  Saved graphics profiles.")

    # Call the new estimation function
    plot_time_to_deorbit(fit_results)
    plot_time_to_deorbit_vs_inclination(fit_results)
    print("\nAll isolated models calculated and graphics saved successfully!")



if __name__ == "__main__":
    df = load_simulation_data()
    if not df.empty:
        analyze_and_plot(df)
    else:
        print("No data to analyze.")