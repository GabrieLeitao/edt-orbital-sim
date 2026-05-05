function dX = tether_dynamics(t, X, params)
% Optimized dynamics engine using Vectorized Lumped Mass Model (LMM)
% X = [r1, r2, ..., rn, v1, v2, ..., vn] (ECI coordinates)

num_masses = length(X) / 6;
pos = reshape(X(1:3*num_masses), 3, num_masses);
vel = reshape(X(3*num_masses+1:end), 3, num_masses);

% 1. Gravity, J2 Perturbation & Atmospheric Drag (Vectorized)
r_norms = sqrt(sum(pos.^2, 1));
r_norms3 = r_norms.^3;
r_norms5 = r_norms3 .* r_norms.^2;

% Basic Gravity
accel = -params.mu ./ r_norms3 .* pos;

% J2 Perturbation
z = pos(3,:);
z2 = z.^2;
r2 = r_norms.^2;
pref = (1.5 * params.J2 * params.mu * params.R_e^2) ./ r_norms5;
a_j2 = pref .* [pos(1,:) .* (5*z2./r2 - 1); ...
                pos(2,:) .* (5*z2./r2 - 1); ...
                pos(3,:) .* (5*z2./r2 - 3)];
accel = accel + a_j2;

% Atmospheric Drag
% Sample environment at each node
for i = 1:num_masses
    env_i = environment(pos(:,i), vel(:,i), t, params);
    % Simplified: assume static atmosphere (v_rel = v)
    v_norm = norm(vel(:,i));
    if i == 1 % Target
        node_area = params.Area_sc; % Assuming similar area for target for simplicity
    elseif i == 2 % SC
        node_area = params.Area_sc;
    else % Beads and Tip
        node_area = (params.L_edt / params.N_edt) * 0.001; % 1mm wire assumed
    end
    f_drag = -0.5 * env_i.rho * params.Cd * node_area * v_norm * vel(:,i);
    accel(:,i) = accel(:,i) + f_drag / params.m_vec(i);
end

% 2. Internal Forces (Tension)
% Link 1: Rope (Target to SC)
r_rope = pos(:,2) - pos(:,1);
v_rope = vel(:,2) - vel(:,1);
L_r = norm(r_rope);
L_dot_r = dot(r_rope, v_rope) / L_r;
T_rope = max(0, params.k_rope * (L_r - params.L_rope) + params.c_rope * L_dot_r);
f_T_rope = (T_rope / L_r) * r_rope;

accel(:,1) = accel(:,1) + f_T_rope / params.m_target;
accel(:,2) = accel(:,2) - f_T_rope / params.m_sc;

% EDT Links: Vectorized
L0_seg = params.L_edt / params.N_edt;
m_vec = params.m_vec'; % 1xN row vector for division

p_a = pos(:, 2:end-1);
p_b = pos(:, 3:end);
v_a = vel(:, 2:end-1);
v_b = vel(:, 3:end);

r_seg = p_b - p_a;
v_seg = v_b - v_a;
L_seg = sqrt(sum(r_seg.^2, 1));
L_dot_seg = sum(r_seg .* v_seg, 1) ./ L_seg;

T_seg = max(0, params.k_edt * (L_seg - L0_seg) + params.c_edt * L_dot_seg);
f_T_seg = (T_seg ./ L_seg) .* r_seg;

% Distribute Tension to nodes
accel(:, 2:end-1) = accel(:, 2:end-1) + f_T_seg ./ m_vec(2:end-1);
accel(:, 3:end) = accel(:, 3:end) - f_T_seg ./ m_vec(3:end);

% 3. Lorentz Force (Applied to EDT segments)
% Midpoint environment sampling
env = environment((p_a + p_b)/2, (v_a + v_b)/2, t, params);
f_L = env.I * cross(r_seg, env.B);

% Distribute Lorentz force to both ends of segment
accel(:, 2:end-1) = accel(:, 2:end-1) + 0.5 * f_L ./ m_vec(2:end-1);
accel(:, 3:end) = accel(:, 3:end) + 0.5 * f_L ./ m_vec(3:end);

% Assemble dX
dX = [X(3*num_masses+1:end); accel(:)];

end
