import PolicyState
import State
import tensorflow as tf
from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n, N_coutries


def betaK(state, policy_state):
    return tf.constant((1.0 - betaL), shape=(state.shape[0],))
    
