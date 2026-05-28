import numpy as np
import os
import hashlib
import threading
import time
from frames import get_rotation_matrix_eci
from integrators import rk45_dopri_integrate, velocity_verlet_integrate, lsoda_integrate, IntegratorSolution
from dynamics import tether_dynamics_fast
from scipy.integrate import solve_ivp

def setup_initial_state(params):
    """
    Sets up the initial state vector for the coupled multi-body system.
    
    Supports:
    - Inclination (params.inc)
    - Eccentricity (params.e)
    
    Mission Configuration: 'Perpendicular Rope'
    1. SC and Target: Same altitude (a_init), separated by L_rope in-track.
    2. EDT: Deployed radially inward from the Spacecraft.
    3. Motion: Velocities initialized for consistency with the orbital plane.
    
    Indices (Radial-Inward):
    - Index 0: Tip Mass
    - Index 1 to N_edt-1: EDT flexible beads
    - Index N_edt: Spacecraft (SC)
    - Index N_edt + 1: Target Satellite
    """
    # 1. Basic Orbital Parameters
    a = params.R_e + params.alt
    e = params.e
    inc = params.inc
    mu = params.mu
    
    # Target at Periapsis for simplicity if e > 0
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu / a * (1.0 + e) / (1.0 - e))
    omega = v_p / r_p # Instantaneous angular velocity at periapsis
    
    num_masses = params.num_masses
    l_rope = params.L_rope
    l_edt = params.L_edt
    
    # 2. Define state in 'Orbital Plane' (X=Radial, Y=In-Track, Z=Cross-Track)
    # This frame has Z along angular momentum.
    pos_orb = np.zeros((num_masses, 3))
    vel_orb = np.zeros((num_masses, 3))
    
    # Target (Index N_edt + 1) at [r_p, 0, 0]
    idx_target = params.N_edt + 1
    pos_orb[idx_target] = np.array([r_p, 0.0, 0.0])
    vel_orb[idx_target] = np.array([0.0, v_p, 0.0])
    
    # Spacecraft (Index N_edt) at [r_p, -L_rope, 0]
    idx_sc = params.N_edt
    pos_orb[idx_sc] = np.array([r_p, -l_rope, 0.0])
    # Velocity includes the 'swing' term for the in-track separation
    vel_orb[idx_sc] = np.array([omega * l_rope, v_p, 0.0])
    
    # Tip (Index 0) at [r_p - L_edt, -L_rope, 0]
    pos_orb[0] = np.array([r_p - l_edt, -l_rope, 0.0])
    vel_orb[0] = np.array([omega * l_rope, omega * (r_p - l_edt), 0.0])
    
    # EDT beads (Index 1 to N_edt-1) distributed along the radial line at y = -L_rope
    l0_seg = l_edt / params.N_edt
    for i in range(1, params.N_edt):
        # distance below SC
        h_below = (params.N_edt - i) * l0_seg
        r_node = r_p - h_below
        pos_orb[i] = np.array([r_node, -l_rope, 0.0])
        vel_orb[i] = np.array([omega * l_rope, omega * r_node, 0.0])
        
    # 3. Rotate from Orbital Plane to ECI
    # Standard transformation using modular Rx(inc) from frames.py.
    # This ensures a prograde orbit (h_z > 0) moving Northward at t=0.
    R_inc = get_rotation_matrix_eci(inc)
    
    pos = np.dot(pos_orb, R_inc.T)
    vel = np.dot(vel_orb, R_inc.T)
    
    X0 = np.zeros(6 * num_masses)
    X0[:3*num_masses] = pos.flatten()
    X0[3*num_masses:] = vel.flatten()
    return X0

def save_checkpoint(run_folder, t, X, p_arr, current_idx=0, total_compute_time=0.0):
    """
    Saves the current state and metadata to a binary checkpoint.
    History is managed separately via memory-mapped files.
    """
    p_hash = hashlib.sha256(p_arr.tobytes()).hexdigest()
    checkpoint_path = os.path.join(run_folder, "checkpoint.npz")
    
    save_args = {
        "t": t, "X": X, "p_arr": p_arr, "p_hash": p_hash,
        "current_idx": current_idx,
        "total_compute_time": total_compute_time
    }
    np.savez(checkpoint_path, **save_args)
    
    with open(os.path.join(run_folder, "last_checkpoint.txt"), "w") as f:
        f.write(f"Last checkpoint saved at t = {t:.2f} s\nParameter Hash: {p_hash}")

def load_checkpoint(run_folder):
    """
    Loads latest state and metadata from binary checkpoint.
    """
    checkpoint_path = os.path.join(run_folder, "checkpoint.npz")
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        p_hash = str(data['p_hash']) if 'p_hash' in data.files else None
        current_idx = int(data['current_idx']) if 'current_idx' in data.files else 0
        total_compute_time = float(data['total_compute_time']) if 'total_compute_time' in data.files else 0.0
        return float(data['t']), data['X'], data['p_arr'], p_hash, current_idx, total_compute_time
    return None, None, None, None, 0, 0.0

def integrate_system(X0, t_span, p_arr, rtol=1e-7, atol=1e-9, pbar=None,
                     sampling_hz=1.0, method='RK45'):
    """
    Pure integration driver. Dispatches to compiled integrators in `integrators.py`,
    keeping the integration loop free of Python crossings. 
    
    Progress is reported via shared memory and a background thread to keep the
    UI responsive without 'breaking' the integrator's internal logic.
    """
    t0, tf = t_span
    span = tf - t0
    method_u = method.upper()

    # Output grid (e.g. 1 Hz telemetry). +1 so endpoint is included.
    n_total = int(span * sampling_hz) + 1
    t_eval = np.linspace(t0, tf, n_total)

    n_state = X0.shape[0]
    Y_out = np.empty((n_total, n_state))
    Y_out[0] = X0

    # Setup shared progress pointer (Slot 27 in p_arr_ext)
    # We use a 28-element array to match integrators.P_ARR_LEN
    p_arr_ext = np.zeros(28)
    p_arr_ext[:len(p_arr)] = p_arr
    p_arr_ext[27] = t0
    
    # Progress Monitor Thread
    stop_event = threading.Event()
    def monitor():
        last_reported_t = t0
        while not stop_event.is_set():
            curr_t = p_arr_ext[27]
            dt = int(curr_t - last_reported_t)
            if dt > 0 and pbar is not None:
                pbar.update(dt)
                last_reported_t += dt
            time.sleep(1.0) # 1Hz update as requested

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    if pbar is not None:
        monitor_thread.start()

    try:
        y = X0.astype(np.float64).copy()
        if method_u == 'RK45':
            # RK45 takes a view of the progress slot
            rk45_dopri_integrate(t0, tf, y, p_arr_ext, t_eval, rtol, atol, Y_out, p_arr_ext[27:28])
        elif method_u == 'VERLET':
            velocity_verlet_integrate(t0, tf, y, p_arr_ext, t_eval, Y_out, p_arr_ext[27:28])
        elif method_u == 'LSODA':
            lsoda_integrate(t0, tf, y, p_arr_ext, t_eval, rtol, atol, Y_out)
        elif method_u == 'RADAU':
            sol = solve_ivp(lambda t, y: tether_dynamics_fast(t, y, p_arr), (t0, tf), y,
                            method='Radau', t_eval=t_eval, rtol=1e-5, atol=1e-7)
            Y_out[:] = sol.y.T
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'RK45', 'VERLET' or 'LSODA'.")
    finally:
        stop_event.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
            # Final catch-up update
            if pbar is not None:
                remaining = int(tf - p_arr_ext[27])
                if remaining > 0:
                    pbar.update(remaining)

    return IntegratorSolution(t=t_eval, y=Y_out.T)
