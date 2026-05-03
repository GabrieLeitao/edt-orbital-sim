import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import os

def interactive_visualization(csv_path=os.path.join("results", "simulation_results.csv")):
    # 1. Load Data
    val_path = os.path.join("results", "validation_results.csv")
    if not os.path.exists(csv_path) and not os.path.exists(val_path):
        print(f"Error: Neither {csv_path} nor {val_path} found. Run simulate.py or validate_physics.py first.")
        return

    # Fallback to validation if main results missing
    actual_path = csv_path if os.path.exists(csv_path) else val_path
    print(f"Loading data from {actual_path}...")
    df = pd.read_csv(actual_path)
    
    pos_cols = [c for c in df.columns if '_rx_m' in c]
    num_masses = len(pos_cols)
    frames = len(df)
    
    data_3d = np.zeros((frames, num_masses, 3))
    for i in range(num_masses):
        col_prefix = pos_cols[i].replace('_rx_m', '')
        data_3d[:, i, 0] = df[f'{col_prefix}_rx_m']
        data_3d[:, i, 1] = df[f'{col_prefix}_ry_m']
        data_3d[:, i, 2] = df[f'{col_prefix}_rz_m']

    # 2. Setup Figure
    fig = plt.figure(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # Subplot 1: Orbital View (Global)
    ax_orbit = fig.add_subplot(1, 2, 1, projection='3d')
    ax_orbit.set_title("Global Orbital View (ECI) [km]")
    
    # Subplot 2: Relative View (Tether Dynamics)
    ax_rel = fig.add_subplot(1, 2, 2, projection='3d')
    ax_rel.set_title("Relative Tether View (Fixed on Target) [m]")

    # 3. Initialize Elements
    target_idx = num_masses - 1
    
    # Global Path (Static for context)
    skip = max(1, frames // 500)
    path = data_3d[::skip, target_idx, :]
    ax_orbit.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', alpha=0.3, label='Orbital Track')
    
    # Current State Objects
    points_orbit, = ax_orbit.plot([], [], [], 'ro', markersize=4)
    line_rel, = ax_rel.plot([], [], [], 'k-o', lw=2, markersize=5)
    
    # Earth Wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    re = 6378137.0
    ax_orbit.plot_wireframe(re*np.cos(u)*np.sin(v), re*np.sin(u)*np.sin(v), re*np.cos(v), color="gray", alpha=0.1)

    # 4. Widget Setup
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time [s]', 0, frames-1, valinit=0, valstep=1)
    
    ax_button = plt.axes([0.05, 0.05, 0.1, 0.04])
    btn_play = Button(ax_button, 'Play/Pause')

    # 5. Interactive Logic
    is_playing = [False]
    
    def update_view(frame_idx):
        frame_idx = int(frame_idx)
        current_pos = data_3d[frame_idx, :, :]
        
        # Orbital View Update (Follow Target)
        points_orbit.set_data(current_pos[:, 0], current_pos[:, 1])
        points_orbit.set_3d_properties(current_pos[:, 2])
        
        cam_dist = 1e6
        ax_orbit.set_xlim3d([current_pos[target_idx, 0] - cam_dist, current_pos[target_idx, 0] + cam_dist])
        ax_orbit.set_ylim3d([current_pos[target_idx, 1] - cam_dist, current_pos[target_idx, 1] + cam_dist])
        ax_orbit.set_zlim3d([current_pos[target_idx, 2] - cam_dist, current_pos[target_idx, 2] + cam_dist])
        
        # Relative View Update
        rel_data = current_pos - current_pos[target_idx]
        line_rel.set_data(rel_data[:, 0], rel_data[:, 1])
        line_rel.set_3d_properties(rel_data[:, 2])
        
        # Reset relative view range
        ax_rel.set_xlim3d([-2500, 2500])
        ax_rel.set_ylim3d([-2500, 2500])
        ax_rel.set_zlim3d([-2500, 2500])
        
        fig.canvas.draw_idle()

    def on_slider_change(val):
        update_view(val)

    def on_play_click(event):
        is_playing[0] = not is_playing[0]

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_click)

    # 6. Animation Loop (Manual)
    def animate_step():
        if is_playing[0]:
            new_val = (slider.val + 1) % frames
            slider.set_val(new_val)
        plt.pause(0.01) # Low pause for interactivity

    update_view(0)
    print("Interactive window opened. Use the slider to scrub through time.")
    
    while plt.fignum_exists(fig.number):
        animate_step()

if __name__ == "__main__":
    interactive_visualization()
