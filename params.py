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
        self.E_rope = 100e9           # Young's Modulus (Kevlar) [Pa]
        self.diam_rope = 0.002        # 2mm rope
        
        # Derived Rope Damping (Targeting zeta = 0.5 for the rope)
        area_rope = np.pi * (self.diam_rope / 2.0)**2
        k_rope = (self.E_rope * area_rope) / self.L_rope
        # Natural frequency w = sqrt(k/m). Using SC mass as reference.
        w_rope = np.sqrt(k_rope / self.m_sc)
        # c = beta * k = 2 * zeta * sqrt(k*m) => beta = 2 * zeta / w
        self.beta_rope = 2.0 * 0.5 / w_rope 

        # 4. EDT Properties (SC-Tip)
        self.L_edt = 500.0            # Total EDT length [m]
        self.N_edt = 10               # Number of segments
        self.E_edt = 70e9             # Young's Modulus (Aluminum) [Pa]
        self.diam_edt = 0.0015        # 1.5mm wire
        self.rho_aluminum = 2700.0    # Aluminum density [kg/m^3]
        
        # Scientific Scaling: Derive mass from geometry
        area_edt = np.pi * (self.diam_edt / 2.0)**2
        self.m_edt_total = self.L_edt * area_edt * self.rho_aluminum
        
        # Electrical Properties
        self.rho_al_res = 2.65e-8     # Aluminum resistivity [Ohm*m]
        self.z_plasma = 100.0         # Plasma contactor impedance [Ohm]
        self.r_load = 500.0           # Load resistance [Ohm]
        
        # Derived EDT Damping (Targeting zeta = 0.7 for critical damping of 'snaps')
        l_seg = self.L_edt / (self.N_edt + 1)
        m_seg = self.m_edt_total / (self.N_edt + 1)
        k_seg = (self.E_edt * area_edt) / l_seg
        w_seg = np.sqrt(k_seg / m_seg)
        self.beta_edt = 2.0 * 0.7 / w_seg

        # 5. Environment Assumptions
        self.Cd = 2.2                 # Drag coefficient
        self.Area_sc = 1.2            # Effective area [m^2] (SC + Panels)

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
            self.L_rope, self.E_rope, self.diam_rope, self.beta_rope,
            self.L_edt, float(self.N_edt), self.m_edt_total,
            self.E_edt, self.diam_edt, self.beta_edt,
            self.rho_al_res, self.z_plasma, self.r_load, self.Cd, self.Area_sc
        ], dtype=np.float64)

# Indices for the flat array
IDX_MU = 0
IDX_RE = 1
IDX_J2 = 2
IDX_M_TARGET = 3
IDX_M_SC = 4
IDX_M_TIP = 5
IDX_L_ROPE = 6
IDX_E_ROPE = 7
IDX_DIAM_ROPE = 8
IDX_BETA_ROPE = 9
IDX_L_EDT = 10
IDX_N_EDT = 11
IDX_M_EDT_TOTAL = 12
IDX_E_EDT = 13
IDX_DIAM_EDT = 14
IDX_BETA_EDT = 15
IDX_RHO_AL = 16
IDX_Z_PLASMA = 17
IDX_R_LOAD = 18
IDX_CD = 19
IDX_AREA = 20
