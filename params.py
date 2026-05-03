import numpy as np

class SimulationParams:
    def __init__(self):
        # 1. Physical Constants
        self.mu = 3.986004418e14      # Earth's gravitational parameter [m^3/s^2]
        self.R_e = 6378137.0          # Earth's mean radius [m]
        self.J2 = 1.08263e-3          # J2 perturbation coefficient

        # 2. System Masses [kg]
        self.m_target = 800.0         # Target satellite mass
        self.m_sc = 100.0             # Our spacecraft mass
        self.m_tip = 20.0             # EDT tip mass (boom/weight)

        # 3. Tether Properties (Rope: Target-SC)
        self.L_rope = 50.0            # Nominal rope length [m]
        self.k_rope = 5e3             # Stretched stiffness [N/m] (Reduced for stability)
        self.c_rope = 1e2             # Damping [N*s/m]

        # 4. EDT Properties (SC-Tip)
        self.L_edt = 2000.0           # Total EDT length [m]
        self.N_edt = 10               # Number of segments
        self.m_edt_total = 10.0       # Total tether mass [kg]
        self.k_edt = 2e3              # Segment stiffness [N/m] (Reduced for stability)
        self.c_edt = 5e1              # Segment damping [N*s/m]
        self.I_edt = 2.0              # Constant current assumption [A]

        # 5. Environment Assumptions
        self.Cd = 2.2                 # Drag coefficient
        self.Area_sc = 2.0            # Effective area [m^2] (SC + Target)

        # 6. Initial Orbit (LEO)
        self.alt = 500e3              # Altitude [m]
        self.inc = np.radians(51.6)   # Inclination [rad]
        self.e = 0.001                # Near-circular eccentricity

    @property
    def num_masses(self):
        return 3 + self.N_edt

    def to_numba_params(self):
        """Returns a flat array for Numba consumption"""
        return np.array([
            self.mu, self.R_e, self.J2,
            self.m_target, self.m_sc, self.m_tip,
            self.L_rope, self.k_rope, self.c_rope,
            self.L_edt, float(self.N_edt), self.m_edt_total,
            self.k_edt, self.c_edt, self.I_edt,
            self.Cd, self.Area_sc
        ], dtype=np.float64)

# Indices for the flat array
IDX_MU = 0
IDX_RE = 1
IDX_J2 = 2
IDX_M_TARGET = 3
IDX_M_SC = 4
IDX_M_TIP = 5
IDX_L_ROPE = 6
IDX_K_ROPE = 7
IDX_C_ROPE = 8
IDX_L_EDT = 9
IDX_N_EDT = 10
IDX_M_EDT_TOTAL = 11
IDX_K_EDT = 12
IDX_C_EDT = 13
IDX_I_EDT = 14
IDX_CD = 15
IDX_AREA = 16
