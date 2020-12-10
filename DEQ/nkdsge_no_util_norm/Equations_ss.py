import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import alpha, beta, b_habit, eps_p, eps_w, phi_w, phi_p, pi_ss, zeta_w, zeta_p, chi, kappa, delta_0, rho_i, phi_pi, phi_y, psi, rho_A, rho_Z, rho_G, rho_nu, rho_psi, s_A, s_Z, s_G, s_nu, s_psi, s_i, s_w, omega, G_gov, F_prod, i_ss, i_LB, division_eps


def equations(state, policy_state):
    
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}

    delta_1 = Definitions.delta_1(state, policy_state)

    #loss_dict['eq_1'] = PolicyState.lambday(policy_state) - 2.62
    loss_dict['eq_1'] = PolicyState.lambday(policy_state) - 1.0

    #loss_dict['eq_2'] = PolicyState.muy(policy_state) - 2.62 
    loss_dict['eq_2'] = PolicyState.muy(policy_state) - 1.0 
    
    #loss_dict['eq_3'] = PolicyState.Cy(policy_state) - 0.0247 
    loss_dict['eq_3'] = PolicyState.Cy(policy_state) - 1.0 

    #loss_dict['eq_4'] = PolicyState.piy(policy_state) - 0.005 
    loss_dict['eq_4'] = PolicyState.piy(policy_state) - 1.0 


    #loss_dict['eq_5'] = PolicyState.pihashy(policy_state) - 0.0176 
    loss_dict['eq_5'] = PolicyState.pihashy(policy_state) - 1.0 


    #loss_dict['eq_6'] = PolicyState.Ry(policy_state) - 0.03
    loss_dict['eq_6'] = PolicyState.Ry(policy_state) - 1.0


    #loss_dict['eq_8'] = PolicyState.uy(policy_state) - 1.0 

    #loss_dict['eq_7'] = PolicyState.Iy(policy_state) - 0.185 
    loss_dict['eq_7'] = PolicyState.Iy(policy_state) - 1.0 

    #loss_dict['eq_8'] = PolicyState.wy(policy_state) - 1.1913 
    loss_dict['eq_8'] = PolicyState.wy(policy_state) - 1.0 

    #loss_dict['eq_9'] = PolicyState.whashy(policy_state) - 1.1969 
    loss_dict['eq_9'] = PolicyState.whashy(policy_state) - 1.0 

    #loss_dict['eq_10'] = PolicyState.h1y(policy_state) - 1.6711 
    loss_dict['eq_10'] = PolicyState.h1y(policy_state) - 1.0 

    #loss_dict['eq_11'] = PolicyState.h2y(policy_state) - 2.8441
    loss_dict['eq_11'] = PolicyState.h2y(policy_state) - 1.0

    #loss_dict['eq_12'] = PolicyState.Ny(policy_state) - 0.4394 
    loss_dict['eq_12'] = PolicyState.Ny(policy_state) - 1.0 

    #loss_dict['eq_15'] = PolicyState.Khaty(policy_state) - 7.47

    #loss_dict['eq_16'] = PolicyState.Ky(policy_state) - 7.47 

    #loss_dict['eq_13'] = PolicyState.mcy(policy_state) - 0.727 
    loss_dict['eq_13'] = PolicyState.mcy(policy_state) - 1.0 

    #loss_dict['eq_14'] = PolicyState.x1y(policy_state) - 5.34 
    loss_dict['eq_14'] = PolicyState.x1y(policy_state) - 1.0 

    #loss_dict['eq_15'] = PolicyState.x2y(policy_state) - 6.60 
    loss_dict['eq_15'] = PolicyState.x2y(policy_state) - 1.0 

    #loss_dict['eq_16'] = PolicyState.Yy(policy_state) - 0.747 
    loss_dict['eq_16'] = PolicyState.Yy(policy_state) - 1.0 

    #loss_dict['eq_17'] = PolicyState.nupy(policy_state) - 1.0006 
    loss_dict['eq_17'] = PolicyState.nupy(policy_state) - 1.0 
    
    return loss_dict
