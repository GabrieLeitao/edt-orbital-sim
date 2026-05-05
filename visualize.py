import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import os
from frames import eci_to_lvlh
import yaml
from params import SimulationParams
import questionary

def select_simulation_folder():
    results_root = "results"
    target_file = "simulation_results.csv"
    
    if not os.path.exists(results_root):
        print(f"Error: Directory '{results_root}' not found.")
        return None

    # Filter folders
    sim_folders = [f for f in os.listdir(results_root) 
                   if os.path.isdir(os.path.join(results_root, f)) 
                   and target_file in os.listdir(os.path.join(results_root, f))]
    sim_folders.sort(reverse=True)

    if not sim_folders:
        print(f"No folders containing '{target_file}' found.")
        return None

    choices = sim_folders
    answer = questionary.select(
        "Select a simulation run to visualize:",
        choices=choices
    ).ask()
    
    return os.path.join(results_root, answer) if answer else None

def interactive_visualization(csv_path=os.path.join("results", "simulation_results.csv")):
    # 1. Select the subfolder
    sim_folder = select_simulation_folder()
    if not sim_folder:
        return

    # 2. Hardcoded path to the simulation data
    csv_path = os.path.join(sim_folder, "simulation_results.csv")

    # 3. Load the data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 4. Optional: Load the parameters used for this specific run
    yaml_files = [f for f in os.listdir(sim_folder) if f.endswith(('.yaml', '.yml'))]
    
    if yaml_files:
        # Take the first YAML file found in the folder
        yaml_path = os.path.join(sim_folder, yaml_files[0])
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Safely extract the run_id we added earlier
        run_id = config.get("metadata", {}).get("run_id", "Unknown Run ID")
        desc = config.get("metadata", {}).get("description", "N/A")
        print(f"Config loaded: {yaml_files[0]}")
        print(f"Run ID: {run_id} | {desc}")
    else:
        print("No YAML configuration file found in this folder.")
        config = None

    params = SimulationParams.from_yaml(yaml_path)
    re = params.R_e
    
    pos_cols = [c for c in df.columns if '_rx_m' in c]
    vel_cols = [c for c in df.columns if '_vx_ms' in c]
    num_masses = len(pos_cols)
    frames = len(df)
    
    data_pos = np.zeros((frames, num_masses, 3))
    data_vel = np.zeros((frames, num_masses, 3))
    for i in range(num_masses):
        col_p = pos_cols[i].replace('_rx_m', '')
        data_pos[:, i, 0] = df[f'{col_p}_rx_m']
        data_pos[:, i, 1] = df[f'{col_p}_ry_m']
        data_pos[:, i, 2] = df[f'{col_p}_rz_m']
        
        col_v = vel_cols[i].replace('_vx_ms', '')
        data_vel[:, i, 0] = df[f'{col_v}_vx_ms']
        data_vel[:, i, 1] = df[f'{col_v}_vy_ms']
        data_vel[:, i, 2] = df[f'{col_v}_vz_ms']

    # 2. Setup Figure (2x2 Grid)
    fig = plt.figure(figsize=(15, 12))
    plt.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.3)
    
    # Subplot 1: Full Orbit (ECI)
    ax_orbit = fig.add_subplot(2, 2, 1, projection='3d')
    ax_orbit.set_title("Full Orbit Trajectory (ECI)")
    
    # Subplot 2: Sat + Target (In-Plane)
    ax_sat_target = fig.add_subplot(2, 2, 2)
    ax_sat_target.set_title("Sat + Target Behavior (In-Plane)")
    ax_sat_target.set_xlabel("In-Track [m]")
    ax_sat_target.set_ylabel("Radial [m]")
    ax_sat_target.grid(True)
    # Ensure 'down' points to Earth by inverting Y-axis if Radial is positive outwards, 
    # but eci_to_lvlh z points to Earth center. So positive Z is 'down'.
    # We want 'down' to be visually down, so we keep Z as Y-axis and label it accordingly.

    # Subplot 3: EDT Behavior (In-Plane)
    ax_edt = fig.add_subplot(2, 2, 3)
    ax_edt.set_title("EDT Full System Behavior (In-Plane)")
    ax_edt.set_xlabel("In-Track [m]")
    ax_edt.set_ylabel("Radial [m]")
    ax_edt.grid(True)

    # Subplot 4: LVLH 3D Perspective
    ax_lvlh_3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax_lvlh_3d.set_title("LVLH 3D Perspective")

    # 3. Initialize Elements
    target_idx = num_masses - 1
    sc_idx = num_masses - 2
    tip_idx = 0

    # Orbit Elements
    skip = max(1, frames // 500)
    # Background Orbit Path
    ax_orbit.plot(data_pos[::skip, target_idx, 0], data_pos[::skip, target_idx, 1], data_pos[::skip, target_idx, 2], 'k--', alpha=0.3, label="Orbit Path")
    # Current Trajectory Trace
    trace_orbit, = ax_orbit.plot([], [], [], 'r-', lw=1.5, alpha=0.6, label="Trace")
    # Current System Marker
    point_orbit, = ax_orbit.plot([], [], [], 'ro', markersize=6, label="Current Pos")
    
    # Earth Visualization
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    ax_orbit.plot_wireframe(re*np.cos(u)*np.sin(v), re*np.sin(u)*np.sin(v), re*np.cos(v), color="blue", alpha=0.05)
    
    # Equator Line
    theta = np.linspace(0, 2*np.pi, 100)
    ax_orbit.plot(re*np.cos(theta), re*np.sin(theta), 0, color="blue", lw=2, alpha=0.3, label="Equator")
    
    # ECI Reference Frame Axes
    ax_orbit.set_xlabel("X (ECI) [m]")
    ax_orbit.set_ylabel("Y (ECI) [m]")
    ax_orbit.set_zlabel("Z (ECI) [m]")
    ax_orbit.view_init(elev=20, azim=45) # Initial view to see inclination
    ax_orbit.legend(loc='upper right', fontsize='small')

    # Sat + Target Elements (2D)
    # Horizontal: In-Track (X), Vertical: Radial (Z)
    line_st_rope, = ax_sat_target.plot([], [], 'k-', lw=2)
    marker_target_st, = ax_sat_target.plot([], [], 'rs', markersize=8, label="Target")
    marker_sc_st, = ax_sat_target.plot([], [], 'bo', markersize=6, label="SC")
    ax_sat_target.legend()

    # EDT Elements (2D)
    line_edt_full, = ax_edt.plot([], [], 'g-', lw=1.5, alpha=0.8)
    marker_tip_edt, = ax_edt.plot([], [], 'mo', markersize=5, label="Tip")
    marker_sc_edt, = ax_edt.plot([], [], 'bo', markersize=5, label="SC")
    marker_target_edt, = ax_edt.plot([], [], 'rs', markersize=7, label="Target")
    ax_edt.legend()

    # 3D LVLH Elements
    line_lvlh_3d, = ax_lvlh_3d.plot([], [], [], 'g-', lw=2)
    marker_target_3d, = ax_lvlh_3d.plot([], [], [], 'rs', markersize=6)
    marker_sc_3d, = ax_lvlh_3d.plot([], [], [], 'bo', markersize=5)

    # 4. Widget Setup
    is_playing = [False]
    speed_factor = 50  # Playback speed: skip N frames per step
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, 'Time Index', 0, frames-1, valinit=0, valstep=1)
    
    ax_button = plt.axes([0.05, 0.02, 0.1, 0.03])
    btn_play = Button(ax_button, 'Play/Pause')

    # 5. Interactive Logic
    def update_view(frame_idx):
        frame_idx = int(frame_idx)
        r_curr = data_pos[frame_idx]
        v_curr = data_vel[frame_idx]
        r_target = r_curr[target_idx]
        v_target = v_curr[target_idx]
        
        # 1. Full Orbit (ECI)
        point_orbit.set_data([r_target[0]], [r_target[1]])
        point_orbit.set_3d_properties([r_target[2]])
        
        # Update Trace (sample more densely as we play faster)
        trace_skip = max(1, frame_idx // 100)
        trace_data = data_pos[:frame_idx+1:trace_skip, target_idx, :]
        trace_orbit.set_data(trace_data[:, 0], trace_data[:, 1])
        trace_orbit.set_3d_properties(trace_data[:, 2])
        
        # Keep ECI view centered on Earth
        limit = 8e6 
        ax_orbit.set_xlim3d([-limit, limit])
        ax_orbit.set_ylim3d([-limit, limit])
        ax_orbit.set_zlim3d([-limit, limit])
        
        # 2. LVLH Transformation
        # r_lvlh: [N, 3] where 0: In-Track, 1: Cross-Track, 2: Radial (points to Earth)
        r_lvlh = eci_to_lvlh(r_curr, v_target, r_target)
        
        # In-Plane View: In-Track (x) vs Radial (z)
        it = r_lvlh[:, 0]
        ct = r_lvlh[:, 1]
        rd = r_lvlh[:, 2] # Positive RD points to Earth center

        # 3. Sat + Target View (Zoomed)
        # Horizontal axis: In-Track (it), Vertical axis: Radial (rd)
        line_st_rope.set_data(it[[sc_idx, target_idx]], rd[[sc_idx, target_idx]])
        marker_target_st.set_data([it[target_idx]], [rd[target_idx]])
        marker_sc_st.set_data([it[sc_idx]], [rd[sc_idx]])
        
        # Zoom on the 50m rope. Target is at (0,0) in LVLH.
        ax_sat_target.set_xlim([-80, 80])
        ax_sat_target.set_ylim([80, -80]) # Inverted: Down is towards Earth (Positive RD)

        # 4. EDT Behavior View
        line_edt_full.set_data(it, rd)
        marker_tip_edt.set_data([it[tip_idx]], [rd[tip_idx]])
        marker_sc_edt.set_data([it[sc_idx]], [rd[sc_idx]])
        marker_target_edt.set_data([it[target_idx]], [rd[target_idx]])
        
        # Zoom on the 2km tether
        ax_edt.set_xlim([-750, 750])
        ax_edt.set_ylim([750, -750]) # Inverted: Down is towards Earth (Positive RD)

        # 5. LVLH 3D View
        line_lvlh_3d.set_data(it, ct)
        line_lvlh_3d.set_3d_properties(rd)
        marker_target_3d.set_data([it[target_idx]], [ct[target_idx]])
        marker_target_3d.set_3d_properties([rd[target_idx]])
        marker_sc_3d.set_data([it[sc_idx]], [ct[sc_idx]])
        marker_sc_3d.set_3d_properties([rd[sc_idx]])
        
        ax_lvlh_3d.set_xlim3d([-750, 750])
        ax_lvlh_3d.set_ylim3d([-750, 750])
        ax_lvlh_3d.set_zlim3d([750, -750])
        ax_lvlh_3d.set_xlabel("In-Track [m]")
        ax_lvlh_3d.set_ylabel("Cross-Track [m]")
        ax_lvlh_3d.set_zlabel("Radial [m]")
        
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    btn_play.on_clicked(lambda e: is_playing.__setitem__(0, not is_playing[0]))

    def animate_step():
        if is_playing[0]:
            next_val = (slider.val + speed_factor) % frames
            slider.set_val(next_val)
        plt.pause(0.01)

    update_view(0)
    while plt.fignum_exists(fig.number):
        animate_step()

if __name__ == "__main__":
    interactive_visualization()
