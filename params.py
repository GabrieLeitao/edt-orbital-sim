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
        self.m_tip = 2.0             # EDT tip mass (boom/weight)

        # 3. Tether Properties (Rope: Target-SC)
        self.L_rope = 25.0            # Nominal rope length [m]
        self.E_rope = 100e9           # Young's Modulus (Kevlar) [Pa]
        self.diam_rope = 0.002        # 2mm rope
        
        # Derived Rope Damping (Targeting zeta = 0.5 for the rope)
        area_rope = np.pi * (self.diam_rope / 2.0)**2
        self.k_rope = (self.E_rope * area_rope) / self.L_rope
        # Natural frequency w = sqrt(k/m). Using SC mass as reference.
        w_rope = np.sqrt(self.k_rope / self.m_sc)
        # c = beta * k = 2 * zeta * sqrt(k*m) => beta = 2 * zeta / w
        self.beta_rope = 2.0 * 0.5 / w_rope

        # 4. EDT Properties (SC-Tip)
        self.L_edt = 200.0            # Total EDT length [m]
        self.N_edt = 5#10               # Number of segments
        self.E_edt = 70e9             # Young's Modulus (Aluminum) [Pa]
        self.diam_edt = 0.0015        # 1.5mm wire
        self.rho_aluminum = 2700.0    # Aluminum density [kg/m^3]
        
        # Scientific Scaling: Derive mass from geometry
        self.area_edt = np.pi * (self.diam_edt / 2.0)**2
        self.m_edt_total = self.L_edt * self.area_edt * self.rho_aluminum
        
        # Derived EDT Damping (Targeting zeta = 0.7 for critical damping of 'snaps')
        l_seg = self.L_edt / (self.N_edt + 1)
        m_seg = self.m_edt_total / (self.N_edt + 1)
        k_seg = (self.E_edt * self.area_edt) / l_seg
        w_seg = np.sqrt(k_seg / m_seg)
        self.beta_edt = 2.0 * 0.7 / w_seg
        
        # Electrical Properties
        self.rho_al_res = 2.65e-8     # Aluminum resistivity [Ohm*m]
        self.z_plasma = 100.0         # Plasma contactor impedance [Ohm]
        self.r_load = 500.0           # Load resistance [Ohm]

        self.r_wire = self.rho_al_res * self.L_edt / self.area_edt 
        self.r_total = self.r_wire + self.z_plasma + self.r_load

        # 5. Environment Assumptions
        self.Cd = 2.2                 # Drag coefficient
        self.Area_sc = 1.2            # Effective area [m^2] (SC + Panels)

        # 6. Initial Orbit (LEO)
        self.alt = 800e3              # Altitude [m]
        self.inc = np.radians(51.6)   # Inclination [rad]
        self.e = 0.001                # Near-circular eccentricity

    @property
    def num_masses(self):
        return 3 + self.N_edt

    def to_dict(self):
        """Converts parameters to a structured dictionary for YAML export."""
        return {
            "physical_constants": {
                "mu": float(self.mu),
                "R_e": float(self.R_e),
                "J2": float(self.J2),
            },
            "system_masses": {
                "m_target": float(self.m_target),
                "m_sc": float(self.m_sc),
                "m_tip": float(self.m_tip),
            },
            "tether_properties": {
                "L_rope": float(self.L_rope),
                "E_rope": float(self.E_rope),
                "diam_rope": float(self.diam_rope),
                "beta_rope": float(self.beta_rope),
                "k_rope": float(self.k_rope),
            },
            "edt_properties": {
                "L_edt": float(self.L_edt),
                "N_edt": int(self.N_edt),
                "E_edt": float(self.E_edt),
                "diam_edt": float(self.diam_edt),
                "rho_aluminum": float(self.rho_aluminum),
                "area_edt": float(self.area_edt),
                "m_edt_total": float(self.m_edt_total),
                "beta_edt": float(self.beta_edt),
            },
            "electrical_properties": {
                "rho_al_res": float(self.rho_al_res),
                "z_plasma": float(self.z_plasma),
                "r_load": float(self.r_load),
                "r_wire": float(self.r_wire),
                "r_total": float(self.r_total),
            },
            "environment_assumptions": {
                "Cd": float(self.Cd),
                "Area_sc": float(self.Area_sc),
            },
            "initial_orbit": {
                "alt": float(self.alt),
                "inc_rad": float(self.inc),
                "e": float(self.e),
            }
        }

    @classmethod
    def from_yaml(cls, filepath):
        """
        Creates a SimulationParams instance loaded with values from a YAML file.
        """
        import yaml
        
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            
        # Extract just the 'parameters' block (ignoring metadata and results)
        # If the file is JUST parameters, fallback to using the whole file
        p_dict = data.get("parameters", data)
        
        if not p_dict:
            raise ValueError(f"No parameters found in {filepath}")

        # Instantiate a new object with base defaults
        obj = cls()
        
        # 1. Physical Constants
        phys = p_dict.get("physical_constants", {})
        obj.mu = phys.get("mu", obj.mu)
        obj.R_e = phys.get("R_e", obj.R_e)
        obj.J2 = phys.get("J2", obj.J2)
        
        # 2. System Masses
        masses = p_dict.get("system_masses", {})
        obj.m_target = masses.get("m_target", obj.m_target)
        obj.m_sc = masses.get("m_sc", obj.m_sc)
        obj.m_tip = masses.get("m_tip", obj.m_tip)
        
        # 3. Rope Properties
        tether = p_dict.get("tether_properties", {})
        obj.L_rope = tether.get("L_rope", obj.L_rope)
        obj.E_rope = tether.get("E_rope", obj.E_rope)
        obj.diam_rope = tether.get("diam_rope", obj.diam_rope)
        obj.beta_rope = tether.get("beta_rope", obj.beta_rope)
        obj.k_rope = (obj.E_rope * np.pi * (obj.diam_rope / 2.0)**2) / obj.L_rope # Recalculate k_rope based on E and L
        
        # 4. EDT Properties
        edt = p_dict.get("edt_properties", {})
        obj.L_edt = edt.get("L_edt", obj.L_edt)
        obj.N_edt = edt.get("N_edt", obj.N_edt)
        obj.E_edt = edt.get("E_edt", obj.E_edt)
        obj.diam_edt = edt.get("diam_edt", obj.diam_edt)
        obj.rho_aluminum = edt.get("rho_aluminum", obj.rho_aluminum)
        obj.m_edt_total = edt.get("m_edt_total", obj.m_edt_total)
        obj.beta_edt = edt.get("beta_edt", obj.beta_edt)
        obj.area_edt = np.pi * (obj.diam_edt / 2.0)**2 # Recalculate area_edt based on diam_edt

        # 5. Electrical Properties
        elec = p_dict.get("electrical_properties", {})
        obj.rho_al_res = elec.get("rho_al_res", obj.rho_al_res)
        obj.z_plasma = elec.get("z_plasma", obj.z_plasma)
        obj.r_load = elec.get("r_load", obj.r_load)
        obj.r_wire = elec.get("r_wire", obj.r_wire)
        obj.r_total = elec.get("r_total", obj.r_total)

        # 6. Environment Assumptions
        env = p_dict.get("environment_assumptions", {})
        obj.Cd = env.get("Cd", obj.Cd)
        obj.Area_sc = env.get("Area_sc", obj.Area_sc)
        
        # 7. Initial Orbit
        orbit = p_dict.get("initial_orbit", {})
        obj.alt = orbit.get("alt", obj.alt)
        obj.inc = orbit.get("inc_rad", obj.inc) # Note: to_dict maps this to 'inc_rad'
        obj.e = orbit.get("e", obj.e)

        return obj

    def to_numba_params(self):
        """Returns a flat array for Numba consumption"""
        return np.array([
            self.mu, self.R_e, self.J2,
            self.m_target, self.m_sc, self.m_tip,
            self.L_rope, self.E_rope, self.diam_rope, self.k_rope, self.beta_rope,
            self.L_edt, float(self.N_edt), self.m_edt_total,
            self.E_edt, self.diam_edt, self.area_edt, self.beta_edt,
            self.rho_al_res, self.z_plasma, self.r_load, self.r_wire, self.r_total, self.Cd, self.Area_sc
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
IDX_K_ROPE = 9
IDX_BETA_ROPE = 10
IDX_L_EDT = 11
IDX_N_EDT = 12
IDX_M_EDT_TOTAL = 13
IDX_E_EDT = 14
IDX_DIAM_EDT = 15
IDX_AREA_EDT = 16
IDX_BETA_EDT = 17
IDX_RHO_AL = 18
IDX_Z_PLASMA = 19
IDX_R_LOAD = 20
IDX_R_WIRE = 21
IDX_R_TOTAL = 22
IDX_CD = 23
IDX_AREA = 24
