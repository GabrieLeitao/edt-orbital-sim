%% Simulation Parameters and Assumptions
% This file centralizes all constants and assumptions for the simulation.

% 1. Physical Constants
params.mu = 3.986004418e14;      % Earth's gravitational parameter [m^3/s^2]
params.R_e = 6378137;            % Earth's mean radius [m]
params.J2 = 1.08263e-3;          % J2 perturbation coefficient

% 2. System Masses [kg]
params.m_target = 800;           % Target satellite mass
params.m_sc = 100;               % Our spacecraft mass
params.m_tip = 20;               % EDT tip mass (boom/weight)

% 3. Tether Properties (Rope: Target-SC)
params.L_rope = 50;              % Nominal rope length [m]
params.k_rope = 1e5;             % Stiffness [N/m]
params.c_rope = 1e3;             % Damping [N*s/m]

% 4. EDT Properties (SC-Tip)
params.L_edt = 2000;             % Total EDT length [m]
params.N_edt = 10;               % Number of segments (beads = N_edt + 1)
params.m_edt_total = 10;         % Total tether mass [kg]
params.k_edt = 5e4;              % Segment stiffness [N/m]
params.c_edt = 5e2;              % Segment damping [N*s/m]
params.I_edt = 2.0;              % Constant current assumption [A] (can be functionalized)

% 5. Environment Assumptions
params.Cd = 2.2;                 % Drag coefficient
params.Area_sc = 0.64;            % Effective area [m^2]
params.B_mag = 3e-5;             % Reference B-field at LEO [Tesla]

% 6. Initial Orbit (LEO)
params.alt = 500e3;              % Altitude [m]
params.inc = deg2rad(51.6);      % Inclination [rad] (ISS-like)
params.e = 0.001;                % Near-circular eccentricity
