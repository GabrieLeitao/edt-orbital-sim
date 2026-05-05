import numpy as np

def eci_to_lvlh(r_eci, v_eci, r_target_eci):
    """
    Transforms a position vector from ECI to LVLH frame centered on a target.
    
    LVLH Definition (Radial-In-Track-Cross-Track):
    - z_lvlh (Radial): Opposite of the target position vector (points toward Earth center).
    - y_lvlh (Cross-Track): Opposite of the angular momentum vector (h = r x v).
    - x_lvlh (In-Track): Completes the right-handed triad (y x z).
    
    Parameters:
    r_eci: [N, 3] positions in ECI
    v_eci: [3] velocity of the target in ECI
    r_target_eci: [3] position of the target in ECI
    
    Returns:
    r_lvlh: [N, 3] positions in LVLH frame [meters]
    """
    # 1. Define LVLH unit vectors based on target state
    r_mag = np.linalg.norm(r_target_eci)
    
    # Radial (Unit vector pointing from target to Earth center)
    u_z = -r_target_eci / r_mag
    
    # Cross-track (Opposite of Angular Momentum direction)
    h = np.cross(r_target_eci, v_eci)
    h_norm = np.linalg.norm(h)
    
    if h_norm < 1e-6: # Handle cases where orbit is near radial or zero angular momentum
        # If h is zero, the cross-track direction is ill-defined.
        # A common fallback is to align u_y with the Z axis if it's not the radial,
        # or a default like [0,1,0] if u_z is [0,0,1].
        # For simplicity, we'll use a default orthogonal vector if h is near zero.
        # A robust fallback might depend on the specific scenario (e.g., landing).
        # Here, we'll try to define a cross-track that's orthogonal to radial and in-track.
        # If r_target is [x,y,z], and u_z is [-x,-y,-z]/norm(r), and velocity is also somewhat aligned
        # it means h is zero.
        # A simple fallback is to use a vector orthogonal to u_z.
        # If u_z is along Z-axis, u_y can be [1,0,0] or [0,1,0].
        # Let's try a default cross-track that's orthogonal to u_z and attempt to be 'in-track' conceptually.
        # A simple and generally safe choice is to find a vector not parallel to u_z.
        # If u_z is [0,0,1], then [1,0,0] is orthogonal.
        # If u_z is [0,1,0], then [1,0,0] is orthogonal.
        # We can define u_y as [0,0,1] if u_z is not [0,0,1], else [1,0,0].
        if np.allclose(u_z, [0,0,1]):
            u_y = np.array([1.0, 0.0, 0.0])
        else:
            u_y = np.array([0.0, 0.0, 1.0])
        # Ensure u_x is still orthogonal
        u_x = np.cross(u_y, u_z)
    else:
        u_y = -h / h_norm
        # In-track (Completes the triad, points generally along velocity)
        u_x = np.cross(u_y, u_z)
    
    # 2. Rotation matrix [ECI -> LVLH]
    R = np.vstack((u_x, u_y, u_z))
    
    # 3. Relative position in ECI
    r_rel_eci = r_eci - r_target_eci
    
    # 4. Rotate to LVLH
    # r_lvlh = R * r_rel_eci
    r_lvlh = np.dot(r_rel_eci, R.T)
    
    return r_lvlh
