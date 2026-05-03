%% Main Simulation Script
clear; clc; close all;

params = get_params();

% 1. Initial State Setup (LVLH aligned initially)
a = params.R_e + params.alt;
v_orb = sqrt(params.mu / a);
omega = v_orb / a;

% Masses: Target(1), SC(2), Beads(3...N+2), Tip(N+3)
num_masses = 3 + params.N_edt; 
X0 = zeros(6 * num_masses, 1);

% Position indices (ECI)
% Assume target is at origin of LVLH (for setup)
% Initial configuration: Target - SC - Tip (Vertical along radial)
r_target = [a; 0; 0];
v_target = [0; v_orb; 0];

% Fill positions and velocities
pos = zeros(3, num_masses);
vel = zeros(3, num_masses);

pos(:,1) = r_target;
vel(:,1) = v_target;

% SC is 'above' target (along R-bar)
pos(:,2) = pos(:,1) + [params.L_rope; 0; 0];
vel(:,2) = vel(:,1) + [0; omega * params.L_rope; 0];

% EDT beads and Tip
L0_seg = params.L_edt / params.N_edt;
for i = 3:num_masses
    dist = params.L_rope + (i-2)*L0_seg;
    pos(:,i) = pos(:,1) + [dist; 0; 0];
    vel(:,i) = vel(:,1) + [0; omega * dist; 0];
end

X0(1:3*num_masses) = pos(:);
X0(3*num_masses+1:end) = vel(:);

% 2. Integration
t_span = [0, 5400 * 2]; % 2 Orbits
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);

fprintf('Starting simulation...\n');
[t, X] = ode113(@(t, X) tether_dynamics(t, X, params), t_span, X0, options);
fprintf('Simulation finished.\n');

% 3. Post-Processing: Semi-major axis decay
pos_sc = X(:, 4:6); % SC position (2nd mass)
r_sc = sqrt(sum(pos_sc.^2, 2));
vel_sc = X(:, 3*num_masses+4 : 3*num_masses+6);
v_sc2 = sum(vel_sc.^2, 2);
energy = v_sc2/2 - params.mu ./ r_sc;
sma = -params.mu ./ (2 * energy);

figure;
plot(t/60, (sma - sma(1))/1e3);
grid on; xlabel('Time [min]'); ylabel('\Delta SMA [km]');
title('Orbital Decay (Semi-Major Axis)');

% 4. Visualization (Simple Snapshot)
figure;
hold on;
snap_idx = length(t);
final_pos = reshape(X(snap_idx, 1:3*num_masses), 3, num_masses);
plot3(final_pos(1,:), final_pos(2,:), final_pos(3,:), '-ok', 'LineWidth', 2);
axis equal; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Final Configuration (ECI)');
