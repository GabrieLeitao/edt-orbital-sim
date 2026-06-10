import numpy as np
from numba import njit
import params as p
from environment import get_environment_optimized
from frames import get_earth_rotation_components

@njit(fastmath=True, nogil=True)
def get_mass_fast(idx, p_arr, num_masses):
    """
    Numba-compatible mass lookup with lumped-mass distribution.
    Supports both SC_ROPE_EDT_TARGET and SC_EDT_TARGET configs.
    """
    is_sc_edt_target = p_arr[p.IDX_SYSTEM_CONFIG] > 0.5
    n_edt = int(p_arr[p.IDX_N_EDT])
    m_edt_seg = p_arr[p.IDX_M_EDT_TOTAL] / n_edt
    
    if is_sc_edt_target:
        # SC_EDT_TARGET: 0=SC, 1..N-1=Beads, N=Target
        if idx == 0:
            return p_arr[p.IDX_M_SC] + 0.5 * m_edt_seg
        elif idx < n_edt:
            return m_edt_seg
        elif idx == n_edt:
            return p_arr[p.IDX_M_TARGET] + 0.5 * m_edt_seg
        else:
            return 0.0
    else:
        # Legacy: 0=Tip, 1..N-1=Beads, N=SC, N+1=Target
        if idx == 0:
            return p_arr[p.IDX_M_TIP] + 0.5 * m_edt_seg
        elif idx < n_edt:
            return m_edt_seg
        elif idx == n_edt:
            return p_arr[p.IDX_M_SC] + 0.5 * m_edt_seg
        elif idx == n_edt + 1:
            return p_arr[p.IDX_M_TARGET]
        else:
            return 0.0

@njit(fastmath=True, nogil=True)
def smooth_tension(dl, dl_dot, k, beta):
    """
    Calculates tension with a smooth transition from slack to taut.
    Uses Rayleigh (proportional) damping: c = beta * k.
    """
    f_elastic = k * dl
    
    # 3. Calculate Damping Force
    f_damping = (beta * k) * dl_dot
    
    # Smooth transition: 
    # Use a sigmoid-like scaling to prevent the 'hammer blow' of sudden tension
    # Transition width is ~10cm (coefficient 50.0) for numerical stability
    scale = 1.0 / (1.0 + np.exp(-50.0 * dl))

    tension = (f_elastic + f_damping) * scale

    # Re-introduce a generous damping cap to prevent numerical "snaps" 
    # but allow significant physical damping. 
    max_d = max(abs(f_elastic) * 50.0, 10.0)
    if abs(f_damping * scale) > max_d:
        tension = (f_elastic + np.sign(f_damping) * max_d) * scale
    
    return max(0.0, tension)

@njit(fastmath=True, nogil=True)
def get_libration_metrics_fast(pos, vel, p_arr):
    """
    Calculates signed libration angle (pitch) and its derivative (theta_dot).
    """
    num_masses = int(p_arr[p.IDX_NUM_MASSES])
    n_edt = int(p_arr[p.IDX_N_EDT])
    is_sc_edt_target = p_arr[p.IDX_SYSTEM_CONFIG] > 0.5
    
    # 1. Calculate Center of Mass (CoM) Position and Velocity
    r_com = np.zeros(3)
    v_com = np.zeros(3)
    total_m = 0.0
    for i in range(num_masses):
        m = get_mass_fast(i, p_arr, num_masses)
        r_com += m * pos[i]
        v_com += m * vel[i]
        total_m += m
    r_com /= total_m
    v_com /= total_m
    
    # 2. Local Vertical Unit Vector (Radial)
    r_mag = np.linalg.norm(r_com)
    u_r = r_com / max(r_mag, 1e-6)
    
    # 3. Orbit Normal (Angular Momentum direction)
    h_vec = np.cross(r_com, v_com)
    h_mag = np.linalg.norm(h_vec)
    if h_mag < 1e-6:
        u_h = np.array([0.0, 0.0, 1.0])
    else:
        u_h = h_vec / h_mag
    
    # 4. In-Track Unit Vector (Forward)
    u_theta = np.cross(u_h, u_r)
    
    # 5. Tether Axis Unit Vector (Tip to Target or SC to Target)
    if is_sc_edt_target:
        idx_start = 0 # SC
        idx_end = n_edt # Target
    else:
        idx_start = 0 # Tip
        idx_end = n_edt + 1 # Target
        
    r_tether = pos[idx_end] - pos[idx_start]
    l_t = np.linalg.norm(r_tether)
    u_t = r_tether / max(l_t, 1e-6)
    
    # 6. Project Tether onto the Orbital Plane
    proj_r = np.dot(u_t, u_r)
    proj_theta = np.dot(u_t, u_theta)
    theta = np.arctan2(proj_theta, proj_r)
    
    # 7. Calculate Theta Dot (Rate of change of libration)
    v_rel = vel[idx_end] - vel[idx_start]
    omega_t_vec = np.cross(r_tether, v_rel) / max(l_t * l_t, 1e-6)
    omega_orb_vec = h_vec / max(r_mag * r_mag, 1e-6)
    theta_dot = np.dot(omega_t_vec - omega_orb_vec, u_h)
    
    return theta, theta_dot

@njit(fastmath=True, nogil=True)
def get_controlled_current(v_total_emf, pos, vel, p_arr):
    """
    Implements closed-loop current control (PD) to limit libration.
    """
    if p_arr[p.IDX_CONTROL_ENABLE] < 0.5:
        return v_total_emf / p_arr[p.IDX_R_TOTAL]

    theta, theta_dot = get_libration_metrics_fast(pos, vel, p_arr)
    theta_limit = p_arr[p.IDX_PITCH_LIMIT]
    k_p = p_arr[p.IDX_K_P]
    k_d = p_arr[p.IDX_K_D]
    v_max = p_arr[p.IDX_V_MAX]

    # Active Control Logic: 
    # If the libration angle exceeds the limit, we apply an active voltage 
    # to counteract the current that is driving the libration.
    v_active = 0.0
    v_sign = v_total_emf / np.sqrt(v_total_emf**2 + 0.1)

    if abs(theta) > theta_limit:
        error_p = theta - (np.sign(theta) * theta_limit)
        v_active += k_p * error_p * v_sign
        
    v_active += k_d * theta_dot * v_sign
    v_active = max(-v_max, min(v_max, v_active))
    
    return (v_total_emf + v_active) / p_arr[p.IDX_R_TOTAL]

@njit(fastmath=True, nogil=True)
def tether_dynamics_fast(t, X, p_arr):
    """
    Numba-JIT optimized core dynamics.
    X: [r0x, r0y, r0z, ..., v0x, v0y, v0z, ...]
    p_arr: Flat parameter array
    """
    is_sc_edt_target = p_arr[p.IDX_SYSTEM_CONFIG] > 0.5
    n_edt = int(p_arr[p.IDX_N_EDT])
    num_masses = int(p_arr[p.IDX_NUM_MASSES])
    dX = np.zeros_like(X)
    
    # Extract positions and velocities
    pos = X[:3*num_masses].reshape((num_masses, 3))
    vel = X[3*num_masses:6*num_masses].reshape((num_masses, 3))

    accel = np.zeros((num_masses, 3))
    
    # 0. Pre-calculate Earth rotation components
    omega_e = 7.2921151467e-5
    theta_g0 = 0.0 
    theta_gmst = (theta_g0 + omega_e * t) % (2 * np.pi)
    cos_tg = np.cos(theta_gmst)
    sin_tg = np.sin(theta_gmst)
    
    # 1. Gravity, J2 & Atmospheric Drag
    mu = p_arr[p.IDX_MU]
    re = p_arr[p.IDX_RE]
    j2 = p_arr[p.IDX_J2]
    cd = p_arr[p.IDX_CD]
    area_sc = p_arr[p.IDX_AREA_SC]
    area_edt = p_arr[p.IDX_AREA_EDT]
    diam_edt = p_arr[p.IDX_DIAM_EDT]
    diam_rope = p_arr[p.IDX_DIAM_ROPE]
    l_rope = p_arr[p.IDX_L_ROPE]
    l0_edt_seg = p_arr[p.IDX_L_EDT] / n_edt
    
    for i in range(num_masses):
        r = pos[i]
        v = vel[i]
        r_norm = np.linalg.norm(r)
        
        # Basic Gravity: -mu/r^3 * r
        a_g = -mu / r_norm**3 * r
        
        # J2
        z2 = r[2]**2
        r2 = r_norm**2
        pref = 1.5 * j2 * mu * re**2 / r_norm**5
        
        accel[i, 0] = a_g[0] + pref * r[0] * (5 * z2 / r2 - 1)
        accel[i, 1] = a_g[1] + pref * r[1] * (5 * z2 / r2 - 1)
        accel[i, 2] = a_g[2] + pref * r[2] * (5 * z2 / r2 - 3)

        # Atmospheric Drag (All nodes)
        _, rho = get_environment_optimized(r, v, t, p_arr, cos_tg, sin_tg)
        
        # Relative wind
        v_rel = v - np.cross(np.array([0.0, 0.0, omega_e]), r)
        v_rel_norm = np.linalg.norm(v_rel)
        
        # Node area logic
        if is_sc_edt_target:
            if i == 0: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt
            elif i < n_edt: node_area = l0_edt_seg * diam_edt
            else: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt
        else:
            if i == 0: node_area = p_arr[p.IDX_AREA_TIP] + 0.5 * l0_edt_seg * diam_edt
            elif i < n_edt: node_area = l0_edt_seg * diam_edt
            elif i == n_edt: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt + 0.5 * l_rope * diam_rope
            else: node_area = area_sc + 0.5 * l_rope * diam_rope

        if v_rel_norm > 1e-3:
            f_drag = -0.5 * rho * cd * node_area * v_rel_norm * v_rel
            accel[i] += f_drag / get_mass_fast(i, p_arr, num_masses)

    # 2. EDT Internal Forces (Tension & Lorentz)
    k_edt_seg = (p_arr[p.IDX_E_EDT] * area_edt) / l0_edt_seg
    beta_edt = p_arr[p.IDX_BETA_EDT]
    
    # PASS 1: Calculate Motional EMF and store B-fields for segments
    v_total_emf = 0.0
    b_fields = np.zeros((n_edt, 3))
    omega_e_vec = np.array([0.0, 0.0, 7.2921151467e-5])
    
    for j in range(n_edt):
        p_a = pos[j]; p_b = pos[j+1]
        r_seg = p_b - p_a
        v_mid = (vel[j] + vel[j+1]) / 2.0
        r_mid = (p_a + p_b) / 2.0
        v_rel_mag = v_mid - np.cross(omega_e_vec, r_mid)
        b_vec, _ = get_environment_optimized(r_mid, v_mid, t, p_arr, cos_tg, sin_tg)
        b_fields[j] = b_vec
        v_total_emf += np.dot(np.cross(v_rel_mag, b_vec), r_seg)
        
    i_dynamic = get_controlled_current(v_total_emf, pos, vel, p_arr)
    
    # PASS 2: Apply Tension and Lorentz Forces
    for j in range(n_edt):
        p_a = pos[j]; p_b = pos[j+1]
        r_seg = p_b - p_a
        v_seg = vel[j+1] - vel[j]
        l_seg = np.linalg.norm(r_seg)
        l_seg_safe = max(l_seg, 1e-6)
        l_dot_seg = np.dot(r_seg, v_seg) / l_seg_safe
        
        t_seg = smooth_tension(l_seg - l0_edt_seg, l_dot_seg, k_edt_seg, beta_edt)
        f_t_seg = (t_seg / l_seg_safe) * r_seg
        
        m_a = get_mass_fast(j, p_arr, num_masses)
        m_b = get_mass_fast(j+1, p_arr, num_masses)
        
        accel[j] += f_t_seg / m_a
        accel[j+1] -= f_t_seg / m_b
        
        # Lorentz Force using pre-calculated B-field
        f_l = i_dynamic * np.cross(r_seg, b_fields[j])
        accel[j] += 0.5 * f_l / m_a
        accel[j+1] += 0.5 * f_l / m_b

    # 3. Rope Link (Only if legacy)
    if not is_sc_edt_target:
        idx_sc = n_edt
        idx_target = n_edt + 1
        r_rope_vec = pos[idx_target] - pos[idx_sc]
        v_rope_vec = vel[idx_target] - vel[idx_sc]
        l_r = np.linalg.norm(r_rope_vec)
        l_r_safe = max(l_r, 1e-6)
        l_dot_r = np.dot(r_rope_vec, v_rope_vec) / l_r_safe
        t_r = smooth_tension(l_r - p_arr[p.IDX_L_ROPE], l_dot_r, p_arr[p.IDX_K_ROPE], p_arr[p.IDX_BETA_ROPE])
        f_t_r = (t_r / l_r_safe) * r_rope_vec
        accel[idx_sc] += f_t_r / get_mass_fast(idx_sc, p_arr, num_masses)
        accel[idx_target] -= f_t_r / get_mass_fast(idx_target, p_arr, num_masses)

    # Assemble dX
    dX[:3*num_masses] = vel.flatten()
    dX[3*num_masses:] = accel.flatten()
    return dX

@njit(fastmath=True)
def compute_physics_metrics(t, X, p_arr):
    """
    Optimized physical metrics calculation.
    """
    is_sc_edt_target = p_arr[p.IDX_SYSTEM_CONFIG] > 0.5
    n_edt = int(p_arr[p.IDX_N_EDT])
    num_masses = int(p_arr[p.IDX_NUM_MASSES])
    
    pos = X[:3*num_masses].reshape((num_masses, 3))
    vel = X[3*num_masses:6*num_masses].reshape((num_masses, 3))
    
    # Earth rotation components
    omega_e = 7.2921151467e-5
    theta_g0 = 0.0 
    theta_gmst = (theta_g0 + omega_e * t) % (2 * np.pi)
    cos_tg = np.cos(theta_gmst)
    sin_tg = np.sin(theta_gmst)

    cd = p_arr[p.IDX_CD]
    area_sc = p_arr[p.IDX_AREA_SC]
    diam_edt = p_arr[p.IDX_DIAM_EDT]
    diam_rope = p_arr[p.IDX_DIAM_ROPE]
    l_rope = p_arr[p.IDX_L_ROPE]
    l0_edt_seg = p_arr[p.IDX_L_EDT] / n_edt
    
    # 1. Drag Calculation
    total_drag_force = np.zeros(3)
    for i in range(num_masses):
        r = pos[i]; v = vel[i]
        _, rho = get_environment_optimized(r, v, t, p_arr, cos_tg, sin_tg)
        
        v_rel = v - np.cross(np.array([0.0, 0.0, omega_e]), r)
        v_rel_norm = np.linalg.norm(v_rel)
        
        if is_sc_edt_target:
            if i == 0 or i == n_edt: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt
            else: node_area = diam_edt * l0_edt_seg
        else:
            if i == n_edt + 1: node_area = area_sc + 0.5 * l_rope * diam_rope
            elif i == n_edt: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt + 0.5 * l_rope * diam_rope
            elif i == 0: node_area = p_arr[p.IDX_AREA_TIP] + 0.5 * l0_edt_seg * diam_edt
            else: node_area = diam_edt * l0_edt_seg
            
        total_drag_force -= 0.5 * rho * cd * node_area * v_rel_norm * v_rel

    # 2. Current and Lorentz Calculation
    v_total_emf = 0.0
    total_lorentz_force = np.zeros(3)
    for j in range(n_edt):
        p_a = pos[j]; p_b = pos[j+1]
        r_seg = p_b - p_a
        v_mid = (vel[j] + vel[j+1]) / 2.0
        r_mid = (p_a + p_b) / 2.0
        
        b_vec, _ = get_environment_optimized(r_mid, v_mid, t, p_arr, cos_tg, sin_tg)
        v_total_emf += np.dot(np.cross(v_mid, b_vec), r_seg)
        
    i_dynamic = get_controlled_current(v_total_emf, pos, vel, p_arr)
    
    for j in range(n_edt):
        p_a, p_b = pos[j], pos[j+1]
        r_seg = p_b - p_a
        b_vec, _ = get_environment_optimized((p_a + p_b)/2.0, (vel[j] + vel[j+1])/2.0, t, p_arr, cos_tg, sin_tg)
        total_lorentz_force += i_dynamic * np.cross(r_seg, b_vec)

    return i_dynamic, np.linalg.norm(total_lorentz_force), np.linalg.norm(total_drag_force)

@njit(fastmath=True)
def tether_jacobian_fast(t, X, p_arr):
    """
    Numba-JIT optimized numerical Jacobian.
    Calculates J = df/dX using finite differences for acceleration terms.
    
    Structure:
    J = [ 0  I ]  (Upper half: d(pos_dot)/d(pos)=0, d(pos_dot)/d(vel)=I)
        [ Jr Jv ] (Lower half: d(vel_dot)/d(pos)=Jr, d(vel_dot)/d(vel)=Jv)
    """
    n = X.shape[0]
    m = n // 2 # Number of position/velocity components (3 * num_masses)
    jac = np.zeros((n, n))
    
    # 1. Upper right block: d(pos_dot)/d(vel) = I
    for i in range(m):
        jac[i, m + i] = 1.0
        
    # 2. Lower blocks: Finite Difference for Accel sensitivities
    # We use a forward-difference for Jr and Jv.
    # To optimize, we mutate X in-place and restore it.
    f0 = tether_dynamics_fast(t, X, p_arr)
    a0 = f0[m:]
    
    # Jr = d(accel)/d(pos)
    for j in range(m):
        # Use relative epsilon for Earth-scale positions (~7,000 km)
        eps = 1e-8 * max(1.0, abs(X[j]))
        orig_val = X[j]
        X[j] += eps
        f_eps = tether_dynamics_fast(t, X, p_arr)
        jac[m:, j] = (f_eps[m:] - a0) / eps
        X[j] = orig_val
        
    # Jv = d(accel)/d(vel)
    for j in range(m):
        # Use relative epsilon for velocities (~7,000 m/s)
        eps = 1e-8 * max(1.0, abs(X[m + j]))
        orig_val = X[m + j]
        X[m + j] += eps
        f_eps = tether_dynamics_fast(t, X, p_arr)
        jac[m:, m + j] = (f_eps[m:] - a0) / eps
        X[m + j] = orig_val
        
    return jac
