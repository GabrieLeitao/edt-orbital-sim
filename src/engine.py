import numpy as np
import os
import hashlib
import threading
import time
from frames import get_rotation_matrix_eci
from integrators import rk45_dopri_integrate, velocity_verlet_integrate, lsoda_integrate, IntegratorSolution
from dynamics import tether_dynamics_fast
from scipy.integrate import solve_ivp
import params as p

def setup_initial_state(params):
    """
    Sets up the initial state vector for the coupled multi-body system.
    
    Initializes the system such that the Center of Mass (CoM) is in a circular orbit
    at the altitude specified by params.alt. This prevents the initial "kick" 
    caused by velocity mismatches across the tether length.

    Supports:
    - Inclination (params.inc)
    - Eccentricity (params.e)
    - System topology (params.system_config) and alignment (params.mission_config)

    System Topologies:
    'SC_ROPE_EDT_TARGET' (legacy): Tip(0)-EDT-SC(N_edt)-rope-Target(N_edt+1)
    'SC_EDT_TARGET' (new): SC(0)-EDT-Target(N_edt), no rope, single chain;
        from Earth outward the order is SC (innermost) -> EDT -> Target (outermost).

    Mission Configurations (alignment of the initial chain):
    'RADIAL' (both topologies):
        All components aligned along the local vertical; Target farthest from
        Earth, EDT tip / SC closest to Earth.
    'PERPENDICULAR' (legacy only):
        SC and Target at the same altitude, separated by L_rope in-track;
        EDT deployed radially inward from the Spacecraft.
    'FULL_IN_TRACK' (SC_EDT_TARGET only):
        The whole SC-EDT-Target chain laid along the velocity (in-track)
        direction; Target leading, SC trailing.

    Indices (legacy, radial-inward):
    - Index 0: Tip Mass
    - Index 1 to N_edt-1: EDT flexible beads
    - Index N_edt: Spacecraft (SC)
    - Index N_edt + 1: Target Satellite
    """
    # 1. Basic Orbital Parameters for System CoM
    a_com = params.R_e + params.alt
    e = params.e
    inc = params.inc
    mu = params.mu
    
    # CoM at Periapsis
    r_p_com = a_com * (1.0 - e)
    v_p_com = np.sqrt(mu / a_com * (1.0 + e) / (1.0 - e))
    omega_com = v_p_com / r_p_com 
    
    num_masses = params.num_masses
    l_rope = params.L_rope
    l_edt = params.L_edt
    n_edt = params.N_edt
    is_sc_edt_target = (params.system_config == 'SC_EDT_TARGET')
    
    # 2. Define relative positions in 'Orbital Plane' (X=Radial, Y=In-Track)
    # Positions are relative to the system CoM initially, then shifted.
    m_nodes = np.zeros(num_masses)
    p_arr_dummy = params.to_numba_params()
    from dynamics import get_mass_fast
    for i in range(num_masses):
        m_nodes[i] = get_mass_fast(i, p_arr_dummy, num_masses)
    total_m = np.sum(m_nodes)
    
    pos_rel = np.zeros((num_masses, 3))
    
    if hasattr(params, 'mission_config') and params.mission_config == 'RADIAL':
        # --- RADIAL ALIGNMENT ---
        # Target (Index N_edt + 1) at farthest: [r_p, 0, 0]
        # SC (Index N_edt) at [r_p - L_rope, 0, 0]
        # Tip (Index 0) at [r_p - L_rope - L_edt, 0, 0]
        
        # Local coords (Radial-Inward from Target):
        y_local = np.zeros(num_masses)
        l0_seg = l_edt / n_edt
        
        if is_sc_edt_target:
            # SC_EDT_TARGET: 0=SC, 1..N-1=Beads, N=Target
            y_local[n_edt] = 0.0 # Target at farthest
            for i in range(n_edt):
                y_local[i] = -(n_edt - i) * l0_seg
        else:
            # Legacy: 0=Tip, 1..N-1=Beads, N=SC, N+1=Target
            y_local[n_edt + 1] = 0.0 # Target at farthest
            y_local[n_edt] = -l_rope
            y_local[0] = -l_rope - l_edt
            for i in range(1, n_edt):
                y_local[i] = -l_rope - (n_edt - i) * l0_seg
            
        y_com_local = np.sum(m_nodes * y_local) / total_m
        for i in range(num_masses):
            pos_rel[i, 0] = y_local[i] - y_com_local
    elif is_sc_edt_target and params.mission_config == 'FULL_IN_TRACK':
        # --- FULL IN-TRACK (SC_EDT_TARGET only) ---
        # The whole single chain is laid along the in-track (velocity) axis.
        # Target leads (s=0), SC trails; x (radial) is 0 for all nodes.
        # SC_EDT_TARGET indexing: 0=SC, 1..N-1=Beads, N=Target.
        s_loc = np.zeros(num_masses)  # in-track coordinate
        l0_seg = l_edt / n_edt
        s_loc[n_edt] = 0.0  # Target leading
        for i in range(n_edt):
            s_loc[i] = -(n_edt - i) * l0_seg

        s_com_loc = np.sum(m_nodes * s_loc) / total_m
        for i in range(num_masses):
            pos_rel[i, 1] = s_loc[i] - s_com_loc
    else:
        # --- PERPENDICULAR (legacy SC_ROPE_EDT_TARGET only) ---
        # SC and Target share an altitude separated by L_rope in-track; the EDT
        # (Tip..SC) hangs radially inward from the SC. x=radial, y=in-track.
        x_loc = np.zeros(num_masses)
        y_loc = np.zeros(num_masses)
        l0_seg = l_edt / n_edt

        x_loc[n_edt + 1] = 0.0
        y_loc[n_edt + 1] = 0.0
        x_loc[n_edt] = 0.0
        y_loc[n_edt] = -l_rope
        x_loc[0] = -l_edt
        y_loc[0] = -l_rope
        for i in range(1, n_edt):
            x_loc[i] = -(n_edt - i) * l0_seg
            y_loc[i] = -l_rope

        x_com_loc = np.sum(m_nodes * x_loc) / total_m
        y_com_loc = np.sum(m_nodes * y_loc) / total_m
        for i in range(num_masses):
            pos_rel[i, 0] = x_loc[i] - x_com_loc
            pos_rel[i, 1] = y_loc[i] - y_com_loc

    pos_orb = np.zeros((num_masses, 3))
    vel_orb = np.zeros((num_masses, 3))
    r_com_vec = np.array([r_p_com, 0.0, 0.0])
    v_com_vec = np.array([0.0, v_p_com, 0.0])
    omega_vec = np.array([0.0, 0.0, omega_com])
    
    for i in range(num_masses):
        pos_orb[i] = r_com_vec + pos_rel[i]
        vel_orb[i] = v_com_vec + np.cross(omega_vec, pos_rel[i])
        
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
    
    Progress is reported via shared memory and a background thread.
    
    Responsiveness Fix: The integrator itself runs in a background thread so the
    main thread remains responsive to SIGINT (Ctrl+C).
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

    # Setup shared progress pointer and Abort Flag
    # We use a 40-element array to match integrators.P_ARR_LEN
    p_arr_ext = np.zeros(40)
    p_arr_ext[:len(p_arr)] = p_arr
    p_arr_ext[p.IDX_PROGRESS] = t0
    p_arr_ext[p.IDX_ABORT] = 0.0 # Abort flag
    
    # Progress Monitor Thread
    stop_event = threading.Event()
    def monitor():
        last_reported_t = t0
        while not stop_event.is_set():
            curr_t = p_arr_ext[p.IDX_PROGRESS]
            dt = int(curr_t - last_reported_t)
            if dt > 0 and pbar is not None:
                pbar.update(dt)
                last_reported_t += dt
            time.sleep(1.0) # 1Hz update as requested

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    if pbar is not None:
        monitor_thread.start()

    # Integrator Thread — capture any exception so the main thread can re-raise it.
    integrator_exc = []
    def run_integrator():
        try:
            y = X0.astype(np.float64).copy()
            if method_u == 'RK45':
                rk45_dopri_integrate(t0, tf, y, p_arr_ext, t_eval, rtol, atol, Y_out)
            elif method_u == 'VERLET':
                velocity_verlet_integrate(t0, tf, y, p_arr_ext, t_eval, Y_out)
            elif method_u == 'LSODA':
                lsoda_integrate(t0, tf, y, p_arr_ext, t_eval, rtol, atol, Y_out)
            elif method_u == 'RADAU':
                sol = solve_ivp(lambda t, y: tether_dynamics_fast(t, y, p_arr), (t0, tf), y,
                                method='Radau', t_eval=t_eval, rtol=1e-5, atol=1e-7)
                Y_out[:] = sol.y.T
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'RK45', 'VERLET' or 'LSODA'.")
        except BaseException as e:
            integrator_exc.append(e)

    it_thread = threading.Thread(target=run_integrator, daemon=True)
    it_thread.start()

    try:
        # Main thread waits in an interruptible loop
        while it_thread.is_alive():
            it_thread.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\nInterrupt received. Stopping simulation gracefully...")
        p_arr_ext[p.IDX_ABORT] = 1.0 # Signal Numba to abort
        it_thread.join()
        raise
    finally:
        stop_event.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
            # Final catch-up update
            if pbar is not None:
                remaining = int(p_arr_ext[p.IDX_PROGRESS] - t0) # Progress made
                # pbar.update is incremental, so we just finish the bar if complete
                pass

    if integrator_exc:
        raise integrator_exc[0]

    return IntegratorSolution(t=t_eval, y=Y_out.T)
