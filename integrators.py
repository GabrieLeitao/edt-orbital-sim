"""
High-performance ODE integrators that keep the integration loop entirely
inside compiled code, eliminating the per-RHS-call Python tax that dominated
the scipy.solve_ivp profile.

Two methods provided:
- 'RK45': adaptive Dormand-Prince 5(4) implemented in pure @njit
- 'VERLET': fixed-step Velocity Verlet (symplectic) implemented in pure @njit
- 'LSODA': numbalsoda's LSODA (compiled C) called via a cfunc RHS wrapper
"""
import numpy as np
import numba as nb
from numba import njit, cfunc

import params as p
from dynamics import tether_dynamics_fast

# numbalsoda is imported lazily inside the LSODA path — it loads compiled
# C/Fortran (~5s) and we don't want to pay that just to use RK45.


# Length of the flat parameter array from SimulationParams.to_numba_params().
# Increased to 28 to include a slot for progress reporting (IDX_PROGRESS_T = 27).
P_ARR_LEN = 28


# ---------------------------------------------------------------------------
# Dormand-Prince 5(4) coefficients (Butcher tableau)
# ---------------------------------------------------------------------------
_C2 = 1.0/5.0;  _C3 = 3.0/10.0; _C4 = 4.0/5.0; _C5 = 8.0/9.0
_A21 = 1.0/5.0
_A31 = 3.0/40.0;     _A32 = 9.0/40.0
_A41 = 44.0/45.0;    _A42 = -56.0/15.0;     _A43 = 32.0/9.0
_A51 = 19372.0/6561.0; _A52 = -25360.0/2187.0; _A53 = 64448.0/6561.0; _A54 = -212.0/729.0
_A61 = 9017.0/3168.0;  _A62 = -355.0/33.0;     _A63 = 46732.0/5247.0; _A64 = 49.0/176.0; _A65 = -5103.0/18656.0
# 5th-order solution weights (same as A7x — last row of tableau)
_B1 = 35.0/384.0; _B3 = 500.0/1113.0; _B4 = 125.0/192.0; _B5 = -2187.0/6784.0; _B6 = 11.0/84.0
# Error estimate (b5 - b4) for adaptive step control
_E1 = 71.0/57600.0; _E3 = -71.0/16695.0; _E4 = 71.0/1920.0; _E5 = -17253.0/339200.0; _E6 = 22.0/525.0; _E7 = -1.0/40.0


@njit(fastmath=True, cache=True, nogil=True)
def rk45_dopri_integrate(t0, tf, y0, p_arr, t_eval, rtol, atol, Y_out, progress_ptr):
    """
    Adaptive Dormand-Prince 5(4) integrator with FSAL.
    Output at t_eval via cubic Hermite interpolation.

    Fills Y_out in-place. Y_out shape (len(t_eval), n_state). 
    Y_out[0] == y0; t_eval[0] must == t0.
    Updates progress_ptr[0] with current t.
    """
    n = y0.shape[0]
    m = t_eval.shape[0]
    # Y_out[0] is already assumed to be y0 from caller

    # Stage buffers
    k1 = np.empty(n); k2 = np.empty(n); k3 = np.empty(n)
    k4 = np.empty(n); k5 = np.empty(n); k6 = np.empty(n); k7 = np.empty(n)
    y = np.empty(n); y_new = np.empty(n); y_tmp = np.empty(n)
    for i in range(n):
        y[i] = y0[i]

    # Physically grounded step size (CFL condition)
    # Speed of sound c_s = sqrt(E / rho)
    # dt_wave = dl / c_s
    l_edt = p_arr[p.IDX_L_EDT]
    n_edt = p_arr[p.IDX_N_EDT]
    e_edt = p_arr[p.IDX_E_EDT]
    rho_al = p_arr[p.IDX_RHO_AL]
    
    dl = l_edt / n_edt
    c_s = np.sqrt(e_edt / rho_al)
    dt_wave = dl / c_s
    
    # Step limits: h_max must resolve wave propagation across one element
    h = 0.1 * dt_wave
    h_max = 0.5 * dt_wave
    h_min = 1e-9

    # Bootstrap k1
    dy = tether_dynamics_fast(t0, y, p_arr)
    for i in range(n):
        k1[i] = dy[i]

    t = t0
    eval_idx = 1
    err_prev = 1.0
    order = 5.0
    alpha = 0.7 / order
    beta = 0.4 / order

    while t < tf and eval_idx < m:
        if t + h > tf:
            h = tf - t
        if h < h_min:
            h = h_min

        # --- 6 stages (k1 already known via FSAL) ---
        for i in range(n): y_tmp[i] = y[i] + h*_A21*k1[i]
        dy = tether_dynamics_fast(t + _C2*h, y_tmp, p_arr)
        for i in range(n): k2[i] = dy[i]

        for i in range(n): y_tmp[i] = y[i] + h*(_A31*k1[i] + _A32*k2[i])
        dy = tether_dynamics_fast(t + _C3*h, y_tmp, p_arr)
        for i in range(n): k3[i] = dy[i]

        for i in range(n): y_tmp[i] = y[i] + h*(_A41*k1[i] + _A42*k2[i] + _A43*k3[i])
        dy = tether_dynamics_fast(t + _C4*h, y_tmp, p_arr)
        for i in range(n): k4[i] = dy[i]

        for i in range(n): y_tmp[i] = y[i] + h*(_A51*k1[i] + _A52*k2[i] + _A53*k3[i] + _A54*k4[i])
        dy = tether_dynamics_fast(t + _C5*h, y_tmp, p_arr)
        for i in range(n): k5[i] = dy[i]

        for i in range(n): y_tmp[i] = y[i] + h*(_A61*k1[i] + _A62*k2[i] + _A63*k3[i] + _A64*k4[i] + _A65*k5[i])
        dy = tether_dynamics_fast(t + h, y_tmp, p_arr)
        for i in range(n): k6[i] = dy[i]

        # 5th-order candidate
        for i in range(n):
            y_new[i] = y[i] + h*(_B1*k1[i] + _B3*k3[i] + _B4*k4[i] + _B5*k5[i] + _B6*k6[i])

        # k7 at endpoint (FSAL): reused as next-step k1 if accepted
        dy = tether_dynamics_fast(t + h, y_new, p_arr)
        for i in range(n): k7[i] = dy[i]

        # Error norm
        err_sq = 0.0
        for i in range(n):
            e_i = h * (_E1*k1[i] + _E3*k3[i] + _E4*k4[i] + _E5*k5[i] + _E6*k6[i] + _E7*k7[i])
            sc = atol + rtol * max(abs(y[i]), abs(y_new[i]))
            r = e_i / sc
            err_sq += r * r
        err_norm = np.sqrt(err_sq / n)

        if err_norm <= 1.0:
            t_new = t + h
            # Cubic Hermite interpolation at any t_eval points in (t, t_new]
            while eval_idx < m and t_eval[eval_idx] <= t_new + 1e-12:
                te = t_eval[eval_idx]
                if te >= t_new - 1e-12:
                    for i in range(n):
                        Y_out[eval_idx, i] = y_new[i]
                else:
                    s = (te - t) / h
                    s2 = s * s
                    s3 = s2 * s
                    h00 = 2.0*s3 - 3.0*s2 + 1.0
                    h10 = s3 - 2.0*s2 + s
                    h01 = -2.0*s3 + 3.0*s2
                    h11 = s3 - s2
                    for i in range(n):
                        Y_out[eval_idx, i] = h00*y[i] + h10*h*k1[i] + h01*y_new[i] + h11*h*k7[i]
                eval_idx += 1

            for i in range(n):
                y[i] = y_new[i]
                k1[i] = k7[i]
            t = t_new
            progress_ptr[0] = t # Memory-based progress report

            if err_norm < 1e-10:
                h_factor = 5.0
            else:
                h_factor = 0.9 * err_norm**(-alpha) * err_prev**beta
            if h_factor > 5.0: h_factor = 5.0
            if h_factor < 0.2: h_factor = 0.2
            h *= h_factor
            if h > h_max: h = h_max
            err_prev = err_norm if err_norm > 1e-4 else 1e-4
        else:
            h_factor = 0.9 * err_norm**(-1.0/order)
            if h_factor < 0.1: h_factor = 0.1
            h *= h_factor
            if h < h_min:
                break

    # Fill any trailing un-evaluated points with the last state (defensive)
    while eval_idx < m:
        for i in range(n):
            Y_out[eval_idx, i] = y[i]
        eval_idx += 1

    return Y_out


@njit(fastmath=True, cache=True, nogil=True)
def velocity_verlet_integrate(t0, tf, y0, p_arr, t_eval, Y_out, progress_ptr):
    """
    Fixed-step Velocity Verlet integrator.
    Symplectic (2nd order) for conservative forces.
    
    Fills Y_out in-place. Y_out shape (len(t_eval), n_state).
    Updates progress_ptr[0] with current t.
    """
    n_state = y0.shape[0]
    m_eval = t_eval.shape[0]
    n_masses = n_state // 6
    n_dof = 3 * n_masses
    
    # Y_out[0] assumed filled by caller
    
    # CFL-based fixed step
    l_edt = p_arr[p.IDX_L_EDT]
    n_edt = p_arr[p.IDX_N_EDT]
    e_edt = p_arr[p.IDX_E_EDT]
    rho_al = p_arr[p.IDX_RHO_AL]
    
    dl = l_edt / n_edt
    c_s = np.sqrt(e_edt / rho_al)
    dt_wave = dl / c_s
    
    # Fixed dt: 0.1 * dt_wave is safe for explicit stability
    dt = 0.1 * dt_wave
    
    t = t0
    y = y0.copy()
    
    # Initial Acceleration
    dx_init = tether_dynamics_fast(t, y, p_arr)
    a_n = dx_init[n_dof:].copy()
    
    eval_idx = 1
    while t < tf:
        if t + dt > tf:
            dt_step = tf - t
        else:
            dt_step = dt
            
        # 1. Half-step velocity
        v_n = y[n_dof:]
        v_half = v_n + 0.5 * a_n * dt_step
        
        # 2. Full-step position
        x_n = y[:n_dof]
        x_next = x_n + v_half * dt_step
        
        # 3. Predictor step for acceleration
        t_next = t + dt_step
        y_pred = np.empty(n_state)
        y_pred[:n_dof] = x_next
        y_pred[n_dof:] = v_half
        
        dx_next = tether_dynamics_fast(t_next, y_pred, p_arr)
        a_next = dx_next[n_dof:]
        
        # 4. Final velocity update
        v_next = v_half + 0.5 * a_next * dt_step
        
        # Update state
        y[:n_dof] = x_next
        y[n_dof:] = v_next
        a_n = a_next 
        t = t_next
        progress_ptr[0] = t
        
        # Dense output at t_eval points
        while eval_idx < m_eval and t_eval[eval_idx] <= t + 1e-12:
            for k in range(n_state):
                Y_out[eval_idx, k] = y[k]
            eval_idx += 1
            
    return Y_out


# ---------------------------------------------------------------------------
# LSODA via numbalsoda
# ---------------------------------------------------------------------------
# numbalsoda import + @cfunc compile are both heavy (~5s combined). Defer both
# until LSODA is actually requested so the RK45 path stays cheap to import.
_LSODA_FUNCPTR = None
_lsoda_driver = None


def _get_lsoda():
    """Lazy: imports numbalsoda and JITs the cfunc RHS on first call."""
    global _LSODA_FUNCPTR, _lsoda_driver
    if _LSODA_FUNCPTR is not None:
        return _LSODA_FUNCPTR, _lsoda_driver

    from numbalsoda import lsoda_sig, lsoda as lsoda_driver

    @cfunc(lsoda_sig, nogil=True)
    def _rhs_lsoda(t, y_ptr, dy_ptr, p_ptr):
        p_arr = nb.carray(p_ptr, (P_ARR_LEN,))
        n_edt = int(p_arr[p.IDX_N_EDT])
        neq = 6 * (2 + n_edt)
        y = nb.carray(y_ptr, (neq,))
        dy = nb.carray(dy_ptr, (neq,))
        
        # Report progress (Slot 27)
        p_arr[27] = t
        
        dx = tether_dynamics_fast(t, y, p_arr)
        for i in range(neq):
            dy[i] = dx[i]

    _LSODA_FUNCPTR = _rhs_lsoda.address
    _lsoda_driver = lsoda_driver
    return _LSODA_FUNCPTR, _lsoda_driver


def lsoda_integrate(t0, tf, y0, p_arr, t_eval, rtol, atol, Y_out):
    """
    Numbalsoda LSODA wrapper. Fills Y_out in-place.
    """
    funcptr, lsoda = _get_lsoda()
    usol, success = lsoda(funcptr, y0, t_eval,
                          data=p_arr, rtol=rtol, atol=atol, mxstep=50000)
    if not success:
        raise RuntimeError(f"LSODA failed in interval [{t0}, {tf}]")
    
    Y_out[:] = usol
    return Y_out


# ---------------------------------------------------------------------------
# scipy-compatible solution object so callers see the familiar .t / .y API
# ---------------------------------------------------------------------------
class IntegratorSolution:
    __slots__ = ("t", "y", "success")
    def __init__(self, t, y, success=True):
        self.t = t           # shape (m,)
        self.y = y           # shape (n_state, m) — scipy convention
        self.success = success
