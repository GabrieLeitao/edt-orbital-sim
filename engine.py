import numpy as np
import os
from tqdm import tqdm
from scipy.integrate import solve_ivp
from dynamics import tether_dynamics_fast, tether_jacobian_fast
import hashlib

def setup_initial_state(params):
    """
    Sets up the initial state vector for the coupled multi-body system.
    
    Mathematical Assumptions:
    1. Radial Alignment: The system is initialized in a 'Gravity Gradient' stable configuration, 
       stretching from the Tip Mass (closest to Earth) to the Target Satellite (highest altitude).
    2. Circular Orbit Approximation: Initial velocities are based on Keplerian circular velocity 
       at the target's altitude, adjusted linearly by the local orbital frequency (omega) 
       across the tether length.
    3. State Vector: $X = [r_0, r_1, ..., r_n, v_0, v_1, ..., v_n]^T$ where $n$ is the number of masses.
    
    Configuration (Radial-Inward):
    - Index 0: Tip Mass (Boom/Stabilizer)
    - Index 1 to N_edt: EDT flexible beads (Lumped Mass Model)
    - Index N_edt + 1: Spacecraft (SC)
    - Index N_edt + 2: Target Satellite
    """
    a_init = params.R_e + params.alt
    v_orb = np.sqrt(params.mu / a_init)
    omega = v_orb / a_init
    num_masses = params.num_masses
    
    pos = np.zeros((num_masses, 3))
    vel = np.zeros((num_masses, 3))
    
    # Reference: Target at the highest point
    r_target = np.array([a_init, 0.0, 0.0])
    v_target = np.array([0.0, v_orb, 0.0])
    
    # Tip (Index 0)
    dist_tip = params.L_edt + params.L_rope
    pos[0] = r_target - np.array([dist_tip, 0.0, 0.0])
    vel[0] = v_target - np.array([0.0, omega * dist_tip, 0.0])
    
    # EDT beads (Index 1 to N_edt)
    # Total segments in EDT chain = N_edt + 1
    L0_seg = params.L_edt / (params.N_edt + 1)
    for i in range(1, params.N_edt + 1):
        # Position is distance from target
        # SC is at L_rope, so first bead is L_rope + L0_seg
        # N_edt-th bead is L_rope + N_edt * L0_seg
        dist = params.L_rope + (params.N_edt + 1 - i) * L0_seg
        pos[i] = r_target - np.array([dist, 0.0, 0.0])
        vel[i] = v_target - np.array([0.0, omega * dist, 0.0])
        
    # Spacecraft (Index N_edt + 1)
    pos[params.N_edt + 1] = r_target - np.array([params.L_rope, 0.0, 0.0])
    vel[params.N_edt + 1] = v_target - np.array([0.0, omega * params.L_rope, 0.0])
    
    # Target (Index N_edt + 2)
    pos[params.N_edt + 2] = r_target
    vel[params.N_edt + 2] = v_target
    
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

def integrate_system(X0, t_span, p_arr, desc, rtol=1e-7, atol=1e-9, pbar=None, sampling_hz=1.0):
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
    sol = solve_ivp(wrapped_dynamics, (t0, tf), X0, method='LSODA', 
                    jac=lambda t, y: tether_jacobian_fast(t, y, p_arr),
                    t_eval=t_eval, rtol=rtol, atol=atol)

    if local_pbar[0] is not None:
        local_pbar[0].close()
    return sol

