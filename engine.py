import numpy as np
import os
from tqdm import tqdm
from scipy.integrate import solve_ivp
from dynamics import tether_dynamics_fast, tether_jacobian_fast
import hashlib

def setup_initial_state(params):
    """
    Sets up the initial state vector for the coupled multi-body system.
    
    Mission Configuration: 'Perpendicular Rope'
    1. SC and Target: Same altitude (a_init), separated by L_rope in-track.
    2. EDT: Deployed radially inward from the Spacecraft.
    3. Motion: Velocities initialized for circular orbit consistency across all nodes.
    
    Indices (Radial-Inward):
    - Index 0: Tip Mass
    - Index 1 to N_edt: EDT flexible beads
    - Index N_edt + 1: Spacecraft (SC)
    - Index N_edt + 2: Target Satellite
    """
    a_init = params.R_e + params.alt
    v_orb = np.sqrt(params.mu / a_init)
    omega = v_orb / a_init
    num_masses = params.num_masses
    l_rope = params.L_rope
    l_edt = params.L_edt
    
    pos = np.zeros((num_masses, 3))
    vel = np.zeros((num_masses, 3))
    
    # Target (Index N_edt + 2) at [a, 0, 0]
    idx_target = params.N_edt + 2
    pos[idx_target] = np.array([a_init, 0.0, 0.0])
    vel[idx_target] = np.array([0.0, omega * a_init, 0.0])
    
    # Spacecraft (Index N_edt + 1) at [a, -L_rope, 0]
    idx_sc = params.N_edt + 1
    pos[idx_sc] = np.array([a_init, -l_rope, 0.0])
    vel[idx_sc] = np.array([omega * l_rope, omega * a_init, 0.0])
    
    # Tip (Index 0) at [a - L_edt, -L_rope, 0]
    pos[0] = np.array([a_init - l_edt, -l_rope, 0.0])
    vel[0] = np.array([omega * l_rope, omega * (a_init - l_edt), 0.0])
    
    # EDT beads (Index 1 to N_edt) distributed along the radial line at y = -L_rope
    l0_seg = l_edt / (params.N_edt + 1)
    for i in range(1, params.N_edt + 1):
        # distance below SC
        h_below = (params.N_edt + 1 - i) * l0_seg
        r_node = a_init - h_below
        pos[i] = np.array([r_node, -l_rope, 0.0])
        vel[i] = np.array([omega * l_rope, omega * r_node, 0.0])
    
    X0 = np.zeros(6 * num_masses)
    X0[:3*num_masses] = pos.flatten()
    X0[3*num_masses:] = vel.flatten()
    return X0

def save_checkpoint(run_folder, t, X, p_arr, history_t=None, history_X=None):
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
        "history_X": history_X if history_X is not None else np.array([])
    }
    np.savez(checkpoint_path, **save_args)
    
    with open(os.path.join(run_folder, "last_checkpoint.txt"), "w") as f:
        f.write(f"Last checkpoint saved at t = {t:.2f} s\nParameter Hash: {p_hash}")

def load_checkpoint(run_folder):
    """
    Loads state and history from binary checkpoint.
    Returns (t, X, p_arr, p_hash, history_t, history_X).
    """
    checkpoint_path = os.path.join(run_folder, "checkpoint.npz")
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        p_hash = str(data['p_hash']) if 'p_hash' in data.files else None
        h_t = data['history_t'] if 'history_t' in data.files else None
        h_X = data['history_X'] if 'history_X' in data.files else None
        return float(data['t']), data['X'], data['p_arr'], p_hash, h_t, h_X
    return None, None, None, None, None, None

def integrate_system(X0, t_span, p_arr, desc, rtol=1e-7, atol=1e-9, pbar=None, sampling_hz=1.0, method='RK45'):
    """
    Driver for the ODE solver with real-time progress feedback.
    Performance: Uses t_eval to downsample output, preventing memory bloat from micro-steps.
    """
    local_pbar = [None]
    t0, tf = t_span
    last_t_rounded = [int(t0)]

    # Downsampling: Ensure we only save state at the requested frequency.
    # 1.0 Hz is scientific standard for long LEO mission telemetry.
    t_eval = np.linspace(t0, tf, int((tf - t0) * sampling_hz) + 1)

    def wrapped_dynamics(t, y):
        # Use provided pbar or create a local one for this segment
        pb = pbar if pbar is not None else local_pbar[0]

        if pb is None and pbar is None:
            local_pbar[0] = tqdm(total=int(tf - t0), unit=' seconds', desc=desc)
            pb = local_pbar[0]

        t_now_rounded = int(t)
        if t_now_rounded > last_t_rounded[0]:
            if pb is not None:
                pb.update(t_now_rounded - last_t_rounded[0])
            last_t_rounded[0] = t_now_rounded
        return tether_dynamics_fast(t, y, p_arr)

    # method='LSODA' handles stiff aluminum EDT dynamics efficiently
    # Providing the jitted Jacobian (jac) significantly speeds up convergence.
    sol = solve_ivp(wrapped_dynamics, (t0, tf), X0, method=method, 
                    # jac=lambda t, y: tether_jacobian_fast(t, y, p_arr),
                    t_eval=t_eval, rtol=rtol, atol=atol)

    if local_pbar[0] is not None:
        local_pbar[0].close()
    return sol

