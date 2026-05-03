import numpy as np
from numba import njit
import params as p
from environment import get_environment_fast

@njit
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

@njit
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
    pos = X[:3*num_masses].reshape((num_masses, 3))
    vel = X[3*num_masses:].reshape((num_masses, 3))
    
    accel = np.zeros((num_masses, 3))
    
    # 1. Gravity, J2 & Atmospheric Drag
    mu = p_arr[p.IDX_MU]
    re = p_arr[p.IDX_RE]
    j2 = p_arr[p.IDX_J2]
    cd = p_arr[p.IDX_CD]
    area = p_arr[p.IDX_AREA]
    
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

        # Atmospheric Drag (Simplified)
        b_vec, rho, i_edt = get_environment_fast(r, v, t, p_arr)
        v_rel = v # Simplified: assume static atmosphere
        v_rel_norm = np.linalg.norm(v_rel)
        if v_rel_norm > 1e-3:
            # Drag is applied to SC and Target (approximate)
            if i >= n_edt + 1:
                f_drag = -0.5 * rho * cd * area * v_rel_norm * v_rel
                accel[i] += f_drag / get_mass_fast(i, p_arr, num_masses)

    # 2. Internal Forces (Tension)
    l0_edt_seg = p_arr[p.IDX_L_EDT] / n_edt
    k_edt = p_arr[p.IDX_K_EDT]
    c_edt = p_arr[p.IDX_C_EDT]
    
    for j in range(n_edt + 1):
        p_a = pos[j]
        p_b = pos[j+1]
        v_a = vel[j]
        v_b = vel[j+1]
        
        r_seg = p_b - p_a
        v_seg = v_b - v_a
        l_seg = np.linalg.norm(r_seg)
        l_seg_safe = max(l_seg, 1e-6)
        l_dot_seg = np.dot(r_seg, v_seg) / l_seg_safe
        
        t_seg = max(0.0, k_edt * (l_seg - l0_edt_seg) + c_edt * l_dot_seg)
        f_t_seg = (t_seg / l_seg_safe) * r_seg
        
        m_a = get_mass_fast(j, p_arr, num_masses)
        m_b = get_mass_fast(j+1, p_arr, num_masses)
        
        accel[j] += f_t_seg / m_a
        accel[j+1] -= f_t_seg / m_b
        
        # 3. Lorentz Force (Applied to EDT segments: Tip to SC)
        b_vec, rho, i_edt = get_environment_fast((p_a + p_b)/2.0, (v_a + v_b)/2.0, t, p_arr)
        f_l = i_edt * np.cross(r_seg, b_vec)
        
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
    
    t_r = max(0.0, p_arr[p.IDX_K_ROPE] * (l_r - p_arr[p.IDX_L_ROPE]) + p_arr[p.IDX_C_ROPE] * l_dot_r)
    f_t_r = (t_r / l_r_safe) * r_rope
    
    accel[idx_sc] += f_t_r / p_arr[p.IDX_M_SC]
    accel[idx_target] -= f_t_r / p_arr[p.IDX_M_TARGET]

    # Assemble dX
    dX[:3*num_masses] = vel.flatten()
    dX[3*num_masses:] = accel.flatten()
    
    return dX
