import numpy as np
import os
from frames import get_rotation_matrix_eci
from integrators import rk45_dopri_integrate, lsoda_integrate, IntegratorSolution
import hashlib

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
    - Index 1 to N_edt: EDT flexible beads
    - Index N_edt + 1: Spacecraft (SC)
    - Index N_edt + 2: Target Satellite
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
    
    # Target (Index N_edt + 2) at [r_p, 0, 0]
    idx_target = params.N_edt + 2
    pos_orb[idx_target] = np.array([r_p, 0.0, 0.0])
    vel_orb[idx_target] = np.array([0.0, v_p, 0.0])
    
    # Spacecraft (Index N_edt + 1) at [r_p, -L_rope, 0]
    idx_sc = params.N_edt + 1
    pos_orb[idx_sc] = np.array([r_p, -l_rope, 0.0])
    # Velocity includes the 'swing' term for the in-track separation
    vel_orb[idx_sc] = np.array([omega * l_rope, v_p, 0.0])
    
    # Tip (Index 0) at [r_p - L_edt, -L_rope, 0]
    pos_orb[0] = np.array([r_p - l_edt, -l_rope, 0.0])
    vel_orb[0] = np.array([omega * l_rope, omega * (r_p - l_edt), 0.0])
    
    # EDT beads (Index 1 to N_edt) distributed along the radial line at y = -L_rope
    l0_seg = l_edt / (params.N_edt + 1)
    for i in range(1, params.N_edt + 1):
        # distance below SC
        h_below = (params.N_edt + 1 - i) * l0_seg
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

def save_checkpoint(run_folder, t, X, p_arr, history_t=None, history_X=None, total_compute_time=0.0):
    """
    Saves the current state and accumulated history to a binary checkpoint.
    Performance: Using .npz for fast binary I/O of history arrays.
    """
    p_hash = hashlib.sha256(p_arr.tobytes()).hexdigest()
    checkpoint_path = os.path.join(run_folder, "checkpoint.npz")
    
    # Pack data. history_t/X are optional for simple state saves.
    save_args = {
        "t": t, "X": X, "p_arr": p_arr, "p_hash": p_hash,
        "history_t": history_t if history_t is not None else np.array([]),
        "history_X": history_X if history_X is not None else np.array([]),
        "total_compute_time": total_compute_time
    }
    np.savez(checkpoint_path, **save_args)
    
    with open(os.path.join(run_folder, "last_checkpoint.txt"), "w") as f:
        f.write(f"Last checkpoint saved at t = {t:.2f} s\nParameter Hash: {p_hash}")

def load_checkpoint(run_folder):
    """
    Loads state and history from binary checkpoint.
    Returns (t, X, p_arr, p_hash, history_t, history_X, total_compute_time).
    """
    checkpoint_path = os.path.join(run_folder, "checkpoint.npz")
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        p_hash = str(data['p_hash']) if 'p_hash' in data.files else None
        h_t = data['history_t'] if 'history_t' in data.files else None
        h_X = data['history_X'] if 'history_X' in data.files else None
        total_compute_time = float(data['total_compute_time']) if 'total_compute_time' in data.files else 0.0
        return float(data['t']), data['X'], data['p_arr'], p_hash, h_t, h_X, total_compute_time
    return None, None, None, None, None, None, 0.0

def integrate_system(X0, t_span, p_arr, rtol=1e-7, atol=1e-9, pbar=None,
                     sampling_hz=1.0, method='RK45', progress_chunk_s=100.0):
    """
    Pure integration driver. Dispatches to compiled integrators in `integrators.py`
    (Numba RK45 / numbalsoda LSODA), keeping the integration loop free of Python
    crossings. If `pbar` is provided, ticks it once per chunk of `progress_chunk_s`
    simulated seconds; otherwise runs silently. Bar ownership is the caller's job.
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

    chunk_pts = max(2, int(progress_chunk_s * sampling_hz))

    y = X0.astype(np.float64).copy()
    i = 0
    while i < n_total - 1:
        j = min(i + chunk_pts, n_total - 1)
        te = t_eval[i:j+1]  # both endpoints, len >= 2

        if method_u == 'RK45':
            Y_chunk = rk45_dopri_integrate(te[0], te[-1], y, p_arr, te, rtol, atol)
        elif method_u == 'LSODA':
            Y_chunk = lsoda_integrate(te[0], te[-1], y, p_arr, te, rtol, atol)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'RK45' or 'LSODA'.")

        Y_out[i:j+1] = Y_chunk
        y = Y_chunk[-1].copy()

        if pbar is not None:
            dt_chunk = int(te[-1]) - int(te[0])
            if dt_chunk > 0:
                pbar.update(dt_chunk)
        i = j

    return IntegratorSolution(t=t_eval, y=Y_out.T)
