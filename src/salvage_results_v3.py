import pandas as pd
import os
import sys

def salvage_all_sequences(run_folder):
    """
    Separates interleaved simulation data by grouping rows based on 
    the tether length (edt_l_m), rounded to the nearest 500m to 
    group jittery data, then sorting each group by time.
    Saves each valid sequence to a separate CSV file.
    """
    file_path = os.path.join(run_folder, "simulation_results.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df = df.dropna(subset=['time_s'])

    # Use tether length as the primary key to separate different simulations
    if 'edt_l_m' not in df.columns:
        print("Column 'edt_l_m' not found. Cannot separate sequences.")
        return
    
    # Robust grouping: round to nearest 500m to group similar tether lengths
    # This ensures 2499, 2500, and 2480 are grouped together.
    df['edt_l_group'] = (df['edt_l_m'] / 500.0).round() * 500.0
    
    # Process each identified sequence
    for edt_group, group in df.groupby('edt_l_group'):
        # Sort and ensure time monotonicity
        group = group.sort_values(by='time_s')
        group = group.drop_duplicates(subset=['time_s'], keep='first')
        
        # Save each group as a separate file
        save_path = os.path.join(run_folder, f"salvaged_edt_group_{int(edt_group)}m.csv")
        group.to_csv(save_path, index=False)
        print(f"Extracted sequence for EDT group ~{int(edt_group)}m: {save_path} ({len(group)} rows)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python salvage_results_v3.py <run_folder>")
        sys.exit(1)
    salvage_all_sequences(sys.argv[1])
