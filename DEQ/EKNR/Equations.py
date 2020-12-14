import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n

def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}
    
   
    betaK = Definitions.betaK(state, policy_state)
    
    
  # original eq 1
    loss_dict['eq_1'] = lambday_norm - (State.nux(state) / (Cy_norm)) + beta * b_habit * E_t(lambda s, ps: State.nux (s) / (Definitions.Cy_norm(s, ps)))      
    
    
    loss_dict['eq_2'] = lambday_norm - beta * (1.0 + iy) * E_t(lambda s, ps: Definitions.lambday_norm(s, ps)*(1.0/(1.0 + Definitions.piy_norm(s, ps))))

    loss_dict['eq_3'] = lambday_norm - muy_norm * State.Zx(state) * ((1.0 - kappa/2.0 * (Iy_norm/State.Ix(state) - 1.0 )**2.0) - kappa * (Iy_norm/State.Ix(state) - 1.0) * Iy_norm/State.Ix(state)) - beta * E_t(lambda s, ps: Definitions.muy_norm(s, ps) * State.Zx(s) * kappa * (Definitions.Iy_norm(s, ps)/Iy_norm - 1.0) * (Definitions.Iy_norm(s, ps)/Iy_norm)**2.0) 

    
    return loss_dict
