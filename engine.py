import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from dynamics import tether_dynamics_fast

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
    L0_seg = params.L_edt / params.N_edt
    for i in range(1, params.N_edt + 1):
        dist = params.L_rope + (params.N_edt - i + 1) * L0_seg
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

def integrate_system(X0, t_span, p_arr, desc, rtol=1e-4, atol=1e-6):
    """
    Driver for the ODE solver with real-time progress feedback.
    """
    pbar_container = [None]
    last_t_rounded = [0]

    def wrapped_dynamics(t, y):
        if pbar_container[0] is None:
            pbar_container[0] = tqdm(total=int(t_span[1]), unit=' seconds', desc=desc)
        t_now_rounded = int(t)
        if t_now_rounded > last_t_rounded[0]:
            pbar_container[0].update(t_now_rounded - last_t_rounded[0])
            last_t_rounded[0] = t_now_rounded
        return tether_dynamics_fast(t, y, p_arr)

    sol = solve_ivp(wrapped_dynamics, t_span, X0, method='LSODA', rtol=rtol, atol=atol)
    
    if pbar_container[0] is not None:
        pbar_container[0].close()
    return sol
