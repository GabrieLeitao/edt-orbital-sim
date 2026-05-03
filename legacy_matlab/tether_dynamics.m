function dX = tether_dynamics(t, X, params)
% Core dynamics engine using Lumped Mass Model (LMM)
% X = [r1, v1, r2, v2, ..., rn, vn] (ECI coordinates)

num_masses = length(X) / 6;
dX = zeros(size(X));

% Extract positions and velocities
pos = reshape(X(1:3*num_masses), 3, num_masses);
vel = reshape(X(3*num_masses+1:end), 3, num_masses);

accel = zeros(3, num_masses);

% 1. Gravity & Perturbations (applied to all)
for i = 1:num_masses
    r = pos(:,i);
    r_norm = norm(r);
    
    % Basic Gravity
    a_g = -params.mu / r_norm^3 * r;
    
    % J2 Perturbation
    z2 = r(3)^2;
    r2 = r_norm^2;
    pref = 1.5 * params.J2 * params.mu * params.R_e^2 / r_norm^5;
    a_j2 = pref * [r(1)*(5*z2/r2 - 1); ...
                   r(2)*(5*z2/r2 - 1); ...
                   r(3)*(5*z2/r2 - 3)];
               
    accel(:,i) = a_g + a_j2;
end

% 2. Internal Forces (Tension)
% Link 1: Target (1) to SC (2)
r_rel = pos(:,2) - pos(:,1);
v_rel = vel(:,2) - vel(:,1);
L = norm(r_rel);
L_dot = dot(r_rel, v_rel) / L;
T = max(0, params.k_rope * (L - params.L_rope) + params.c_rope * L_dot);
f_T = (T / L) * r_rel;
accel(:,1) = accel(:,1) + f_T / params.m_target;
accel(:,2) = accel(:,2) - f_T / params.m_sc;

% EDT Links: SC (2) to Beads (3...N+2) to Tip (N+3)
L0_seg = params.L_edt / params.N_edt;
m_seg = params.m_edt_total / params.N_edt;

for j = 2:(num_masses-1)
    p_a = pos(:,j);   % Current mass
    p_b = pos(:,j+1); % Next mass
    v_a = vel(:,j);
    v_b = vel(:,j+1);
    
    r_seg = p_b - p_a;
    v_seg = v_b - v_a;
    L_seg = norm(r_seg);
    L_dot_seg = dot(r_seg, v_seg) / L_seg;
    
    T_seg = max(0, params.k_edt * (L_seg - L0_seg) + params.c_edt * L_dot_seg);
    f_T_seg = (T_seg / L_seg) * r_seg;
    
    m_a = get_mass(j, params);
    m_b = get_mass(j+1, params);
    
    accel(:,j) = accel(:,j) + f_T_seg / m_a;
    accel(:,j+1) = accel(:,j+1) - f_T_seg / m_b;
    
    % 3. Lorentz Force (Applied to EDT segments)
    % Assumption: Applied at the midpoint of segments between SC and Tip
    if j >= 2
        env = environment((p_a + p_b)/2, (v_a + v_b)/2, t, params);
        f_L = env.I * cross(r_seg, env.B);
        % Distribute Lorentz force to both ends of segment
        accel(:,j) = accel(:,j) + 0.5 * f_L / m_a;
        accel(:,j+1) = accel(:,j+1) + 0.5 * f_L / m_b;
    end
end

% Assemble dX
dX(1:3*num_masses) = X(3*num_masses+1:end);
dX(3*num_masses+1:end) = accel(:);

end

function m = get_mass(idx, params)
    if idx == 1, m = params.m_target;
    elseif idx == 2, m = params.m_sc;
    elseif idx == length(2 + params.N_edt + 1), m = params.m_tip; % Tip is last
    else, m = params.m_edt_total / params.N_edt; % Inner EDT beads
    end
end
