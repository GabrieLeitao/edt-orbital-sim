function env = environment(r, v, t, params)
% Modular environment function for Magnetic Field and Atmosphere (Vectorized)

% 1. Magnetic Field (B)
r_norm = sqrt(sum(r.^2, 1));
u_r = r ./ r_norm;

% Simple Z-aligned dipole approximation
B_0 = 3.12e-5; % Tesla at equator
% B calculation vectorized for 3xN input
B = B_0 * (params.R_e ./ r_norm).^3 .* (3 * u_r(3,:) .* u_r - [0;0;1]);
env.B = B;

% 2. Atmospheric Density (rho)
% Assumption: Exponential model
h = r_norm - params.R_e;
rho_0 = 1.225; % Sea level [kg/m^3]
H = 8500;      % Scale height [m]
env.rho = rho_0 * exp(-h/H);

% 3. Current (I)
env.I = params.I_edt;

end
