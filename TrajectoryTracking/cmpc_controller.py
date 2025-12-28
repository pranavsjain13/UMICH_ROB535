import numpy as np
import cvxpy as cp

def calc_Jacobian(x, u, param):

    L_f = param["L_f"]
    L_r = param["L_r"]
    dt   = param["h"]

    psi = x[2]
    v   = x[3]
    delta = u[1]
    a   = u[0]

    # Jacobian of the system dynamics
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # A = ...
    # B = ...
    beta = np.arctan((L_r / (L_f + L_r)) * np.arctan(delta)) # Slip Angle

    A[0,0] = 1
    A[1,1] = 1
    A[2,2] = 1
    A[3,3] = 1
    A[0, 2] = -v * np.sin(psi + beta) * dt
    A[0, 3] = np.cos(psi + beta) * dt
    A[1, 2] =  v * np.cos(psi + beta) * dt
    A[1, 3] = np.sin(psi + beta) * dt
    A[2, 3] = (dt * np.arctan(delta)) / (np.sqrt((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1) * (L_f + L_r))
    
    B[0, 1] = -dt * L_r * v * np.sin(psi + beta) / ((delta**2 + 1) * np.sqrt((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1) * (L_f + L_r))
    B[1, 1] =  dt * L_r * v * np.cos(psi + beta) / ((delta**2 + 1) * np.sqrt((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1) * (L_f + L_r))
    B[2, 1] =  dt * v / ((delta**2 + 1) * np.sqrt((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1)**(3/2) * (L_f + L_r))
    B[3, 0] = dt

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return [A, B]

def LQR_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl  # input constraints

    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # define the parameters
    Q  = np.eye(4) * 1
    R  = np.eye(2)  * 1
    Pt = np.eye(4) * 10

    # define the cost function
    P = np.zeros((n_var, n_var))
    q = np.zeros((n_var,))

    for k in range(len_state - 1):
        P[k * dim_state:(k+1) * dim_state, k * dim_state:(k+1) * dim_state] = 2.0 * Q # Taken from example
    P[n_x - dim_state:n_x, n_x - dim_state:n_x] = 2.0 * Pt # Taken from example

    for k in range(len_ctrl):
        P[n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = 2.0 * R # Taken from example
    
    # define the constraints    
    A = np.eye( dim_state, n_var)
    b = x0 - x_bar[0, :]

    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)

        Ai = np.zeros((dim_state, n_var))
        Ai[:, k*dim_state:(k+1)*dim_state] = -A_k                   # x[k] term
        Ai[:, (k+1)*dim_state:(k+2)*dim_state] = np.eye(dim_state)  # x[k+1] term
        Ai[:, n_x + k*dim_ctrl:n_x + (k+1)*dim_ctrl] = -B_k         # u[k] term

        A = np.vstack([A, Ai])
        b = np.hstack([b, np.zeros(dim_state)])
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
                      [A @ x == b])
    prob.solve(verbose=False, max_iter = 10000)


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act

def CMPC_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl # input constraints

    a_limit = param["a_lim"]
    delta_limit = param["delta_lim"]
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # define the parameters
    # Q = np.eye(4)  * 5
    # R = np.eye(2)  * 1
    # Pt = np.eye(4) * 10
    Q = np.diag([15.0, 15.0, 8.0, 3.0])
    R = np.diag([0.05, 0.8]) 
    Pt = np.diag([25.0, 25.0, 15.0, 5.0])
    
    # define the cost function
    P = np.zeros((n_var, n_var))
    q = np.zeros(n_var,)

    for k in range(len_state - 1):
        P[k*dim_state:(k+1)*dim_state, k*dim_state:(k+1)*dim_state] = 2.0 * Q # Taken from example
    P[n_x - dim_state:n_x, n_x - dim_state:n_x] = 2.0 * Pt # Taken from example
    for k in range(len_ctrl):
        P[n_x + k*dim_ctrl:n_x + (k+1)*dim_ctrl, n_x + k*dim_ctrl:n_x + (k+1)*dim_ctrl] = 2.0 * R # Taken from example
    
    # define the constraints
    A = np.zeros((dim_state, n_var))
    A[:, :dim_state] = np.eye(dim_state)
    b = x0 - x_bar[0, :]
    G = np.zeros((n_ieq, n_var))
    ub = np.zeros(n_ieq)
    lb = np.zeros(n_ieq)
    
    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)

        Ai = np.zeros((dim_state, n_var))
        Ai[:, k*dim_state:(k+1)*dim_state] = -A_k
        Ai[:, (k+1)*dim_state:(k+2)*dim_state] = np.eye(dim_state)
        Ai[:, n_x + k*dim_ctrl:n_x + (k+1)*dim_ctrl] = -B_k

        A = np.vstack([A, Ai])
        b = np.hstack([b, np.zeros(dim_state)])

        G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = np.eye(dim_ctrl) # Taken from example
        lb[k * dim_ctrl:(k+1) * dim_ctrl] = np.array([a_limit[0], delta_limit[0]]) - u_bar[k, :] # Taken from example
        ub[k * dim_ctrl:(k+1) * dim_ctrl] = np.array([a_limit[1], delta_limit[1]]) - u_bar[k, :] # Taken from example

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    # constraints = [A @ x == b, x >= lb, x <= ub]
    constraints = [A @ x == b, lb <= G @ x, G @ x <= ub]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
    prob.solve(verbose=False, max_iter = 30000)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act