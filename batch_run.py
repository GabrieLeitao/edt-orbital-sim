import subprocess
import os
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define your parameter spaces
lengths = [1000.0, 1500.0, 2000.0]
masses = [300.0, 400.0, 600.0]
inclinations = [98]
# lengths = [500.0, 1000.0, 2000.0, 1500.0, 2500.0]
# masses = [300.0, 400.0, 600.0]
# inclinations = [0.0, 28.5, 51.6]

scenarios = []

# scenarios = [
#     {"target_mass": 400.0, "edt_length": 2500.0, "inclination": 0.0, "system_config": "SC_EDT_TARGET", "mission_config": "RADIAL"},
#     {"target_mass": 300.0, "edt_length": 2500.0, "inclination": 0.0, "system_config": "SC_EDT_TARGET", "mission_config": "RADIAL"},
#     {"target_mass": 400.0, "edt_length": 1500.0, "inclination": 0.0, "system_config": "SC_EDT_TARGET", "mission_config": "RADIAL"},
# ]

completed_runs = {
    (300.0, 1500.0, 51.6),
    (300.0, 1500.0, 28.5),
    (300.0, 1500.0, 0.0),
    (300.0, 1500.0, 87.0),
    (400.0, 1000.0, 51.6),
    (300.0, 1000.0, 51.6),
    (600.0, 1000.0, 51.6),
    (300.0, 1000.0, 87.0),
    (600.0, 1000.0, 28.5),
    (400.0, 1000.0, 28.5),
    (600.0, 1000.0, 87.0),
    (400.0, 1000.0, 0.0),
    (400.0, 1000.0, 87.0),
    (300.0, 1000.0, 28.5),
    (300.0, 1000.0, 0.0),
}

# Generate every permutation using itertools.product
for length, mass, inc in itertools.product(lengths, masses, inclinations):
    if (mass, length, inc) in completed_runs:  # Note: order matches how you stored it
        continue  # Skip the one you already simulated
        
    scenarios.append({
        "target_mass": mass,
        "edt_length": length,
        "inclination": inc,
        "system_config": "SC_EDT_TARGET",
        "mission_config": "RADIAL"
    })

def run_sim(s):
    # Using sys.executable guarantees the workers use the exact same python 
    # environment/virtualenv that you used to start this main script
    cmd = [sys.executable, "src/simulate.py",
           "--target-mass", str(s["target_mass"]),
           "--edt-length", str(s["edt_length"]),
           "--inclination", str(s["inclination"]),
           "--system-config", s["system_config"],
           "--mission-config", s["mission_config"]]
    
    # Run the process safely
    result = subprocess.run(cmd, capture_output=True, text=True)
    return s, result.returncode, result.stdout, result.stderr

if __name__ == "__main__":
    # Ensure results directory exists once before workers start
    os.makedirs('results', exist_ok=True)
    
    # Calculate parallel workers safely
    num_workers = max(1, os.cpu_count() - 1) 
    print(f"Generated {len(scenarios)} total scenarios.")
    print(f"Launching simulations across {min(len(scenarios), num_workers)} background workers...\n")

    # Use ProcessPoolExecutor for robust, crash-free background threading
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the background executor queue
        futures = {executor.submit(run_sim, s): s for s in scenarios}
        
        # As each background simulation finishes, grab its results cleanly
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            s, returncode, stdout, stderr = future.result()
            
            # Print a neat status update for every single completed run
            if returncode == 0:
                print(f"[{completed_count}/{len(scenarios)}] ✅ Success: Mass={s['target_mass']}kg, Length={s['edt_length']}m, Inc={s['inclination']}°")
            else:
                print(f"[{completed_count}/{len(scenarios)}] ❌ Failed: Mass={s['target_mass']}kg, Length={s['edt_length']}m, Inc={s['inclination']}°")
                if stderr:
                    print(f"   Error log: {stderr.strip()}")
        
    print("\nAll simulations completed successfully!")