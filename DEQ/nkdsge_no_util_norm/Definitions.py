import PolicyState
import State
import tensorflow as tf
from Parameters import alpha, beta, b_habit, eps_p, eps_w, phi_w, phi_p, pi_ss, zeta_w, zeta_p, chi, kappa, delta_0, rho_i, phi_pi, phi_y, psi, rho_A, rho_Z, rho_G, rho_nu, rho_psi, s_A, s_Z, s_G, s_nu, s_psi, s_i, s_w, omega, G_gov, F_prod, i_ss, i_LB, division_eps, lambday_ss, muy_ss, Cy_ss, piy_ss, pihashy_ss, Ry_ss, Iy_ss, wy_ss, whashy_ss, h1y_ss, h2y_ss, Ny_ss, x1y_ss, x2y_ss, Yy_ss, nupy_ss, mcy_ss

def delta_1(state, policy_state):
    ##derived quantities / delta_1 =  0.030025125628141 
    return tf.constant(1.0/beta - (1.0 - delta_0), shape=(state.shape[0],))
    
def Ky(state, policy_state):
    ## former equation 14
    return State.Zx(state) * (1.0 - kappa/2.0 * (Iy_norm(state, policy_state)/State.Ix(state) - 1.0 )**2.0) * Iy_norm(state, policy_state) + (1.0 - delta_0) * State.Kx(state)

def iy(state, policy_state):
    ## former equation 19
    return tf.math.maximum((1.0 - rho_i) * i_ss + rho_i * State.ix(state) +  (1.0 - rho_i) * (phi_pi * (piy_norm(state, policy_state) - pi_ss) + phi_y * (tf.math.log(Yy_norm(state, policy_state)) - tf.math.log(State.Yx(state))) + State.mx(state)), i_LB )



### normalized policies
def lambday_norm(state, policy_state):
    return PolicyState.lambday(policy_state)*lambday_ss

def muy_norm(state, policy_state):
    return PolicyState.muy(policy_state)*muy_ss

def Cy_norm(state, policy_state):
    return PolicyState.Cy(policy_state)*Cy_ss

def piy_norm(state, policy_state):
    return PolicyState.piy(policy_state)*piy_ss

def pihashy_norm(state, policy_state):
    return PolicyState.pihashy(policy_state)*pihashy_ss

def Ry_norm(state, policy_state):
    return PolicyState.Ry(policy_state)*Ry_ss

def Iy_norm(state, policy_state):
    return PolicyState.Iy(policy_state)*Iy_ss

def wy_norm(state, policy_state):
    return PolicyState.wy(policy_state)*wy_ss

def whashy_norm(state, policy_state):
    return PolicyState.whashy(policy_state)*whashy_ss

def h1y_norm(state, policy_state):
    return PolicyState.h1y(policy_state)*h1y_ss

def h2y_norm(state, policy_state):
    return PolicyState.h1y(policy_state)*h2y_ss

def Ny_norm(state, policy_state):
    return PolicyState.Ny(policy_state)*Ny_ss

def mcy_norm(state, policy_state):
    return PolicyState.mcy(policy_state)*mcy_ss

def x1y_norm(state, policy_state):
    return PolicyState.x1y(policy_state)*x1y_ss

def x2y_norm(state, policy_state):
    return PolicyState.x1y(policy_state)*x2y_ss

def Yy_norm(state, policy_state):
    return PolicyState.Yy(policy_state)*Yy_ss

def nupy_norm(state, policy_state):
    return PolicyState.nupy(policy_state)*nupy_ss
    
