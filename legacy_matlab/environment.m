function env = environment(r, v, t, params)
% Modular environment function for Magnetic Field and Atmosphere

% 1. Magnetic Field (B)
% Assumption: Simple Tilted Dipole model or Constant B-field for KISS.
% Here we implement a simple dipole-like vector.
r_norm = norm(r);
u_r = r / r_norm;
% Simple Z-aligned dipole approximation
B_0 = 3.12e-5; % Tesla at equator
B = B_0 * (params.R_e / r_norm)^3 * (3 * u_r(3) * u_r - [0;0;1]);
env.B = B;

% 2. Atmospheric Density (rho)
% Assumption: Exponential model
h = r_norm - params.R_e;
rho_0 = 1.225; % Sea level [kg/m^3]
H = 8500;      % Scale height [m]
env.rho = rho_0 * exp(-h/H);

% 3. Current (I)
% Assumption: Constant current for now, but modular for future OML models.
env.I = params.I_edt;

end
