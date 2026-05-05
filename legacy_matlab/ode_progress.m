function status = ode_progress(t, y, flag, t_span)
% Terminal-based inplace progress bar for ODE solvers
% Usage: options = odeset('OutputFcn', @(t,y,flag) ode_progress(t,y,flag,t_span));

    persistent last_progress
    if isempty(last_progress) || strcmp(flag, 'init')
        last_progress = -1;
    end
    
    if strcmp(flag, 'init')
        fprintf('Simulation Progress: %5.1f%%', 0.0);
    elseif strcmp(flag, 'done')
        fprintf('\rSimulation Progress: 100.0%%\n');
    elseif isempty(flag)
        % ODE solvers can pass multiple time steps in a single call
        t_curr = t(end);
        t_start = t_span(1);
        t_end = t_span(end);
        
        progress = 100 * (t_curr - t_start) / (t_end - t_start);
        
        % Only update if progress has changed significantly to reduce I/O
        if abs(progress - last_progress) >= 0.1
            fprintf('\rSimulation Progress: %5.1f%%', progress);
            last_progress = progress;
        end
    end
    status = 0;
end
