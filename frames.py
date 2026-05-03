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
    u_y = -h / np.linalg.norm(h)
    
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
