import PolicyState
import State
import tensorflow as tf
from scipy.special import gamma, factorial
from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n, N_coutries


def betaK(state, policy_state):
    return tf.constant((1.0 - betaL), shape=(state.shape[0],))

def gamma_param(state = None, policy_state = None):
    assert theta > sigma - 1.0, "theta > sigma - 1.0 does not hold"
    arg = (theta - sigma + 1.0)/theta
    gamma_val = (gamma(arg))**(1.0/(sigma - 1.0))
    return tf.constant(gamma_val, shape=(state.shape[0],))

def omega_n(state = None, policy_state = None):
    return tf.constant(1.0/N_coutries, shape=(state.shape[0],))



    
