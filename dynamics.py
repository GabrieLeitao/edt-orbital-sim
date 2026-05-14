import numpy as np
from numba import njit
import params as p
from environment import get_environment_fast

@njit(fastmath=True)
def get_mass_fast(idx, p_arr, num_masses):
    """Numba-compatible mass lookup"""
    n_edt = int(p_arr[p.IDX_N_EDT])
    if idx == 0:
        return p_arr[p.IDX_M_TIP]
    elif idx == n_edt + 1:
        return p_arr[p.IDX_M_SC]
    elif idx == n_edt + 2:
        return p_arr[p.IDX_M_TARGET]
    else:
        return p_arr[p.IDX_M_EDT_TOTAL] / n_edt

@njit(fastmath=True)
def smooth_tension(dl, dl_dot, k, beta):
    """
    Calculates tension with a smooth transition from slack to taut.
    Uses Rayleigh (proportional) damping: c = beta * k.
    
    Mathematical Physics:
    A 'hard' max(0, tension) creates a discontinuity in the force derivative 
    which triggers numerical 'bouncing'. This smooth-slack model uses a 
    sigmoid-like transition to simulate the microscopic 'tightening' of 
    molecular bonds before full tension is reached.
    """
    # Pure linear tension + proportional damping
    # c = beta * k
    # 2. Calculate Elastic Force
    f_elastic = k * dl
    
    # 3. Calculate Damping Force
    f_damping = (beta * k) * dl_dot
    
    # Smooth transition: 
    # Use a sigmoid-like scaling to prevent the 'hammer blow' of sudden tension
    # Transition width is ~10cm (coefficient 50.0) for numerical stability
    scale = 1.0 / (1.0 + np.exp(-50.0 * dl))

    tension = (f_elastic + f_damping) * scale

    if dl > 0:
        max_allowed_damping = abs(f_elastic) * 5.0 # Allow some overshoot, but not infinite
        if abs(f_damping * scale) > max_allowed_damping:
            # Re-calculate tension with capped damping
            d_sign = 1.0 if f_damping > 0 else -1.0
            tension = (f_elastic + (d_sign * max_allowed_damping)) * scale
    
    return max(0.0, tension)

@njit(fastmath=True)
def tether_dynamics_fast(t, X, p_arr):
    """
    Numba-JIT optimized core dynamics.
    X: [r0x, r0y, r0z, ..., v0x, v0y, v0z, ...]
    p_arr: Flat parameter array
    """
    n_edt = int(p_arr[p.IDX_N_EDT])
    num_masses = 3 + n_edt
    dX = np.zeros_like(X)
    
    # Extract positions and velocities
    X_slice = np.ascontiguousarray(X[:3*num_masses])
    pos = X_slice.reshape((num_masses, 3))
    X_slice = np.ascontiguousarray(X[3*num_masses:])
    vel = X_slice.reshape((num_masses, 3))

    accel = np.zeros((num_masses, 3))
    
    # 0. Pre-calculate Earth rotation components once per call
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
    area = p_arr[p.IDX_AREA]
    area_edt = p_arr[p.IDX_AREA_EDT]
    diam_edt = p_arr[p.IDX_DIAM_EDT]
    diam_rope = p_arr[p.IDX_DIAM_ROPE]
    l_rope = p_arr[p.IDX_L_ROPE]
    l0_edt_seg = p_arr[p.IDX_L_EDT] / (n_edt + 1)
    
    from environment import get_environment_optimized
    
    for i in range(num_masses):
        r = pos[i]
        v = vel[i]
        r_norm = np.linalg.norm(r)
        
        # Basic Gravity: -mu/r^3 * r
        a_g = -mu / r_norm**3 * r
        
        # J2 Perturbation
        z = r[2]
        z2 = z**2
        r2 = r_norm**2
        pref = 1.5 * j2 * mu * re**2 / r_norm**5
        
        accel[i, 0] = a_g[0] + pref * r[0] * (5 * z2 / r2 - 1)
        accel[i, 1] = a_g[1] + pref * r[1] * (5 * z2 / r2 - 1)
        accel[i, 2] = a_g[2] + pref * r[2] * (5 * z2 / r2 - 3)

        # Atmospheric Drag (All nodes)
        # Optimized: only need rho for drag, but B is used for Lorentz later.
        # However, for simplicity and to match the physics, we get both.
        _, rho = get_environment_optimized(r, v, t, p_arr, cos_tg, sin_tg)
        
        # Relative wind
        v_rel = v - np.cross(np.array([0.0, 0.0, omega_e]), r)
        v_rel_norm = np.linalg.norm(v_rel)
        
        # Calculate node-specific area
        if i == 0: # Tip
            node_area = p_arr[p.IDX_AREA_TIP] + 0.5 * l0_edt_seg * diam_edt
        elif i <= n_edt: # Beads
            node_area = l0_edt_seg * diam_edt
        elif i == n_edt + 1: # SC
            node_area = area + 0.5 * l0_edt_seg * diam_edt + 0.5 * l_rope * diam_rope
        else: # Target
            node_area = area + 0.5 * l_rope * diam_rope

        if v_rel_norm > 1e-3:
            f_drag = -0.5 * rho * cd * node_area * v_rel_norm * v_rel
            accel[i] += f_drag / get_mass_fast(i, p_arr, num_masses)

    # 2. Internal Forces (Tension) - EDT (SC-Tip)
    k_edt_seg = (p_arr[p.IDX_E_EDT] * area_edt) / l0_edt_seg
    beta_edt = p_arr[p.IDX_BETA_EDT]
    
    # PASS 1: Calculate Motional EMF and store B-fields for segments
    v_total_emf = 0.0
    b_fields = np.zeros((n_edt + 1, 3))
    for j in range(n_edt + 1):
        p_a = pos[j]; p_b = pos[j+1]
        v_a = vel[j]; v_b = vel[j+1]
        
        r_seg = p_b - p_a
        v_mid = (v_a + v_b) / 2.0
        r_mid = (p_a + p_b) / 2.0
        
        b_vec, _ = get_environment_optimized(r_mid, v_mid, t, p_arr, cos_tg, sin_tg)
        b_fields[j] = b_vec
        
        v_induced = np.dot(np.cross(v_mid, b_vec), r_seg)
        v_total_emf += v_induced
        
    i_dynamic = v_total_emf / p_arr[p.IDX_R_TOTAL]
    
    # PASS 2: Apply Tension and Lorentz Forces
    for j in range(n_edt + 1):
        p_a = pos[j]; p_b = pos[j+1]
        v_a = vel[j]; v_b = vel[j+1]
        
        r_seg = p_b - p_a
        v_seg = v_b - v_a
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

    # Rope Link: SC (N+1) to Target (N+2)
    idx_sc = n_edt + 1
    idx_target = n_edt + 2
    r_rope = pos[idx_target] - pos[idx_sc]
    v_rope = vel[idx_target] - vel[idx_sc]
    l_r = np.linalg.norm(r_rope)
    l_r_safe = max(l_r, 1e-6)
    l_dot_r = np.dot(r_rope, v_rope) / l_r_safe
    
    t_r = smooth_tension(l_r - p_arr[p.IDX_L_ROPE], l_dot_r, p_arr[p.IDX_K_ROPE], p_arr[p.IDX_BETA_ROPE])
    f_t_r = (t_r / l_r_safe) * r_rope
    
    accel[idx_sc] += f_t_r / p_arr[p.IDX_M_SC]
    accel[idx_target] -= f_t_r / p_arr[p.IDX_M_TARGET]

    # Assemble dX
    dX[:3*num_masses] = vel.flatten()
    dX[3*num_masses:] = accel.flatten()
    
    return dX

@njit(fastmath=True)
def compute_physics_metrics(t, X, p_arr):
    """
    Optimized physical metrics calculation.
    """
    n_edt = int(p_arr[p.IDX_N_EDT])
    num_masses = 3 + n_edt
    
    X_pos_slice = np.ascontiguousarray(X[:3*num_masses])
    pos = X_pos_slice.reshape((num_masses, 3))
    X_vel_slice = np.ascontiguousarray(X[3*num_masses:])
    vel = X_vel_slice.reshape((num_masses, 3))
    
    # Earth rotation components
    omega_e = 7.2921151467e-5
    theta_g0 = 0.0 
    theta_gmst = (theta_g0 + omega_e * t) % (2 * np.pi)
    cos_tg = np.cos(theta_gmst)
    sin_tg = np.sin(theta_gmst)
    
    from environment import get_environment_optimized

    cd = p_arr[p.IDX_CD]
    area_sc = p_arr[p.IDX_AREA]
    diam_edt = p_arr[p.IDX_DIAM_EDT]
    diam_rope = p_arr[p.IDX_DIAM_ROPE]
    l_rope = p_arr[p.IDX_L_ROPE]
    l0_edt_seg = p_arr[p.IDX_L_EDT] / (n_edt + 1)
    
    # 1. Drag Calculation
    total_drag_force = np.zeros(3)
    for i in range(num_masses):
        r = pos[i]; v = vel[i]
        _, rho = get_environment_optimized(r, v, t, p_arr, cos_tg, sin_tg)
        
        v_rel = v - np.cross(np.array([0.0, 0.0, omega_e]), r)
        v_rel_norm = np.linalg.norm(v_rel)
        
        if i == n_edt + 2: node_area = area_sc + 0.5 * l_rope * diam_rope
        elif i == n_edt + 1: node_area = area_sc + 0.5 * l0_edt_seg * diam_edt + 0.5 * l_rope * diam_rope
        elif i == 0: node_area = p_arr[p.IDX_AREA_TIP] + 0.5 * l0_edt_seg * diam_edt
        else: node_area = diam_edt * l0_edt_seg
            
        total_drag_force -= 0.5 * rho * cd * node_area * v_rel_norm * v_rel

    # 2. Current and Lorentz Calculation
    v_total_emf = 0.0
    total_lorentz_force = np.zeros(3)
    for j in range(n_edt + 1):
        p_a = pos[j]; p_b = pos[j+1]
        r_seg = p_b - p_a
        v_mid = (vel[j] + vel[j+1]) / 2.0
        r_mid = (p_a + p_b) / 2.0
        
        b_vec, _ = get_environment_optimized(r_mid, v_mid, t, p_arr, cos_tg, sin_tg)
        v_total_emf += np.dot(np.cross(v_mid, b_vec), r_seg)
        
    i_dynamic = v_total_emf / p_arr[p.IDX_R_TOTAL]
    
    for j in range(n_edt + 1):
        p_a = pos[j]; p_b = pos[j+1]
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
