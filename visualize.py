import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import os
from frames import eci_to_lvlh

def interactive_visualization(csv_path=os.path.join("results", "simulation_results.csv")):
    # 1. Load Data
    val_path = os.path.join("results", "validation_results.csv")
    if not os.path.exists(csv_path) and not os.path.exists(val_path):
        print(f"Error: Neither {csv_path} nor {val_path} found. Run simulate.py or validate_physics.py first.")
        return

    actual_path = csv_path if os.path.exists(csv_path) else val_path
    print(f"Loading data from {actual_path}...")
    df = pd.read_csv(actual_path)
    
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

    # 2. Setup Figure (3 Panels)
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # Subplot 1: Orbital View (Global ECI)
    ax_orbit = fig.add_subplot(1, 3, 1, projection='3d')
    ax_orbit.set_title("Global Orbital View (ECI)")
    
    # Subplot 2: Local Tether View (Relative ECI)
    ax_rel = fig.add_subplot(1, 3, 2, projection='3d')
    ax_rel.set_title("Relative View (ECI Orientation)")

    # Subplot 3: Technical LVLH View (Radial-InTrack-CrossTrack)
    ax_lvlh = fig.add_subplot(1, 3, 3, projection='3d')
    ax_lvlh.set_title("LVLH Stabilization View (Technical)")

    # 3. Initialize Elements
    target_idx = num_masses - 1
    
    # Orbital Track
    skip = max(1, frames // 500)
    ax_orbit.plot(data_pos[::skip, target_idx, 0], data_pos[::skip, target_idx, 1], data_pos[::skip, target_idx, 2], 'b-', alpha=0.2)
    
    points_orbit, = ax_orbit.plot([], [], [], 'ro', markersize=4)
    line_rel, = ax_rel.plot([], [], [], 'k-o', lw=2, markersize=5)
    line_lvlh, = ax_lvlh.plot([], [], [], 'g-o', lw=2, markersize=5)
    
    # Earth Wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    re = 6378137.0
    ax_orbit.plot_wireframe(re*np.cos(u)*np.sin(v), re*np.sin(u)*np.sin(v), re*np.cos(v), color="gray", alpha=0.05)

    # 4. Widget Setup
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Time [s]', 0, frames-1, valinit=0, valstep=1)
    ax_button = plt.axes([0.05, 0.05, 0.1, 0.04])
    btn_play = Button(ax_button, 'Play/Pause')

    # 5. Interactive Logic
    is_playing = [False]
    
    def update_view(frame_idx):
        frame_idx = int(frame_idx)
        r_curr = data_pos[frame_idx]
        v_curr = data_vel[frame_idx]
        r_target = r_curr[target_idx]
        v_target = v_curr[target_idx]
        
        # 1. Global ECI
        points_orbit.set_data(r_curr[:, 0], r_curr[:, 1])
        points_orbit.set_3d_properties(r_curr[:, 2])
        cam_dist = 8e5
        ax_orbit.set_xlim3d([r_target[0] - cam_dist, r_target[0] + cam_dist])
        ax_orbit.set_ylim3d([r_target[1] - cam_dist, r_target[1] + cam_dist])
        ax_orbit.set_zlim3d([r_target[2] - cam_dist, r_target[2] + cam_dist])
        
        # 2. Relative ECI (Fixed on target, rotating orientation)
        r_rel_eci = r_curr - r_target
        line_rel.set_data(r_rel_eci[:, 0], r_rel_eci[:, 1])
        line_rel.set_3d_properties(r_rel_eci[:, 2])
        ax_rel.set_xlim3d([-2500, 2500]); ax_rel.set_ylim3d([-2500, 2500]); ax_rel.set_zlim3d([-2500, 2500])
        ax_rel.set_xlabel('X [m]'); ax_rel.set_ylabel('Y [m]'); ax_rel.set_zlabel('Z [m]')

        # 3. LVLH (Radial stabilized)
        r_lvlh = eci_to_lvlh(r_curr, v_target, r_target)
        line_lvlh.set_data(r_lvlh[:, 0], r_lvlh[:, 1])
        line_lvlh.set_3d_properties(r_lvlh[:, 2])
        ax_lvlh.set_xlim3d([-2500, 2500]); ax_lvlh.set_ylim3d([-2500, 2500]); ax_lvlh.set_zlim3d([-2500, 2500])
        ax_lvlh.set_xlabel('In-Track [m]'); ax_lvlh.set_ylabel('Cross-Track [m]'); ax_lvlh.set_zlabel('Radial [m]')
        
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    btn_play.on_clicked(lambda e: is_playing.__setitem__(0, not is_playing[0]))

    def animate_step():
        if is_playing[0]:
            slider.set_val((slider.val + 1) % frames)
        plt.pause(0.01)

    update_view(0)
    while plt.fignum_exists(fig.number):
        animate_step()

if __name__ == "__main__":
    interactive_visualization()
