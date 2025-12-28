import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    T = 3.0 # TODO: planning horizon in seconds, controls how far into the future we plan
    N = 30 # TODO: number of control intervals, defines the resolution of our control actions
    h = T / N

    # system dimensions
    Dim_state = 4 # TODO: Number of states: x-position, y-position, orientation, speed
    Dim_ctrl  = 2 # TODO: Number of control inputs: acceleration and steering angle

    # additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1))  # initial condition, # the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1))       # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')                    # desired speed for the ego vehicle
    delta_last = ca.MX.sym('delta_last')          # last steering angle (used for rate constraints)
    params = ca.vertcat(x_init, v_leader, v_des, delta_last)
    
    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0 # Car parameters, do not change
    L_r = 1.0 # Car parameters, do not change

    beta = ca.atan(L_r/ (L_r + L_f) * ca.tan(u_model[1])) # TODO: The angle at which the car moves sideways relative to its orientation

    xdot = ca.vertcat(x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0],
                      x_model[3] * ca.sin(x_model[2] + beta) - v_leader[1], 
                      x_model[3] / L_r * ca.sin(beta),
                      u_model[0]) # TODO: xdot describes how each state variable changes over time based on current state and control (x-position change, y-position change, orientation change, speed change)

    # Discrete time dynmamics model
    Func_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, params], [xdot * h + x_model]) # TODO 
    
    # Declare model variables, note the dimension
    x = ca.MX.sym('x', Dim_state, N + 1) # TODO
    u = ca.MX.sym('u', Dim_ctrl, N) # TODO

    # Define the cost function (objective) components
    # These encourage the car to stay in its lane, follow the leader, and achieve desired speed
    P = (x_model[1] - 2.0)**2 * 1000.0 + (x_model[2])**2 * 100.0 + (x_model[3] - v_des)**2 * 1500.0 # TODO
    L = (x_model[1] - 2.0)**2 * 1.0 + (x_model[2])**2 * 1.0 + (x_model[3] - v_des)**2 * 150.0 + u_model[0]**2 * 0.001 + u_model[1]**2 * 0.0001 # TODO

    Func_cost_terminal = ca.Function('P', [x_model, params], [P])
    Func_cost_running = ca.Function('Q', [x_model, u_model, params], [L])

    # state and control constraints
    state_ub = np.array([200.0, 5.0, 2.0, 80.0]) # TODO: Example: large bounds for position, tighter on lateral position
    state_lb = np.array([-200.0, -5.0, -2.0, 0.0]) # TODO 
    ctrl_ub  = np.array([4.0, 0.6]) # TODO: Control limits for acceleration and steering angle
    ctrl_lb  = np.array([-10.0, -0.6]) # TODO 
    
    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N + 1), 1)))) # TODO, 1)), ub_x.reshape((# TODO, 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N + 1), 1)))) # TODO, 1)), lb_x.reshape((# TODO, 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    # This enforces the system's discrete dynamics, meaning each next state is based on the current state and control.
    cons_dynamics = []
    ub_dynamics = np.zeros((Dim_state * N, 1)) # TODO, 1))
    lb_dynamics = np.zeros((Dim_state * N, 1)) # TODO, 1))
    for k in range(N):
        # Fx represents the calculated state at the next time step based on the dynamics model.
        # For each state variable (e.g., x-position, y-position, orientation, speed), we add a constraint.
        # This loop means that the computed next state (Fx) matches the predicted state (x[:, k+1]).
        Fx = Func_dynmaics_dt(x[:, k], u[:, k], params)  
        # TODO
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] - Fx[j])


    # state constraints: G(x) <= 0
    cons_state = []
    for k in range(N):
        #### collision avoidance:
        rx, ry = 30.0, 2.0
        cons_state.append(1 - (x[0, k]/rx)**2 - (x[1, k]/ry)**2) # TODO)

        #### Maximum lateral acceleration ####
        dx = (x[:, k+1] - x[:, k]) / h  # Change in state over time step
        ay = dx[2] * x[3, k] # TODO: Compute the lateral acc (change in orientation * speed) using the hints
        # ay = x[3, k] * (x[3, k] / L_r * ca.sin(beta)) # TODO: Compute the lateral acc (change in orientation * speed) using the hints
        
        gmu = (0.5 * 0.6 * 9.81)
        # Upper and lower bound on lateral acceleration
        cons_state.append(ay - gmu) # TODO: Define upper bound on lateral acceleration)
        cons_state.append(-ay - gmu) # TODO: Define lower bound on lateral acceleration)

        #### lane keeping ####
        # Upper and lower bound on lateral position
        y_L, y_R = 3.0, -1.0
        cons_state.append(x[1, k] - y_L) # TODO)
        cons_state.append(y_R - x[1, k]) # TODO)

        #### steering rate ####
        delta_rate_max = 0.6
        if k >= 1:
            d_delta = u[1, k] - u[1, k-1] # TODO: Difference between current and previous steering angle 

            # Constraint steering rate to ensure smooth changes, scaled by time step `h` for discretization.
            # Upper and lower bound on steering rate
            cons_state.append(d_delta - delta_rate_max * h) # TODO)
            cons_state.append(-d_delta - delta_rate_max * h) # TODO)
        else:
            d_delta = u[1, k] - delta_last # TODO: for the first input, given d_last from param
            cons_state.append(d_delta - delta_rate_max * h) # TODO)
            cons_state.append(-d_delta - delta_rate_max * h) # TODO)

    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # cost function: # NOTE: You can also hard code everything here
    J = Func_cost_terminal(x[:, -1], params)
    for k in range(N):
        J = J + Func_cost_running(x[:, k], u[:, k], params)

    # initial condition as parameters
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))
    
    # Define variables for NLP solver
    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)))
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_state_cons, ub_init_cons))

    # Create an NLP solver
    prob = {"x": vars_NLP, "p":params, "f": J, "g":cons_NLP}
    
    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], params.shape[0], lb_var, ub_var, lb_cons, ub_cons