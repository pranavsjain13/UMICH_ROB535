import numpy as np

def ACC_Controller(t, x, param):
    vd = param["vd"] # Desired Speed
    v0 = param["v0"] # Lead vehicle velocity
    m = param["m"] # car mass
    Cag = param["Cag"] # Maximum Acceleration
    Cdg = param["Cdg"] # Maximum Deceleration

    # cost function and constraints for the QP
    P = np.zeros((2,2))
    q = np.zeros([2, 1])
    A = np.zeros([5, 2])
    b = np.zeros([5])
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # set the parameters
    lam = 5.0 # CLF decay rate
    alpha = 1.0 # CBF decay rate
    w = 10.0 # slack weight

    D,v = x

    # construct the cost function
    P = np.array([[2.0, 0.0],
                 [0.0, 2.0*w]])
    q = np.array([-2*m*(vd - v), 0.0])
    
    # construct the constraints
    h = 0.5*(v - vd)**2 # Tracking Performance : Control Lyapunov Function (CLF)
    B = D - 1.8*v - 0.5 * np.clip(v - v0, 0, np.inf)**2 / Cdg # Control Barrier Function (CBF)

    A[0,:] = [(v - vd)/m, -1.0]
    A[1,:] = [ (1.0/m)*(1.8 + max(v - v0, 0.0)/Cdg), 0.0 ]
    A[2,:] = [1.0, 0.0]
    A[3,:] = [-1.0, 0.0]
    A[4,:] = [0.0, -1.0]
    
    b[0] = -lam * h
    b[1] = (v0 - v) + alpha * B
    b[2] = m * Cag
    b[3] = m * Cdg
    b[4] = 0.0

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    return A, b, P, q