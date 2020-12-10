import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import alpha, beta, b_habit, eps_p, eps_w, phi_w, phi_p, pi_ss, zeta_w, zeta_p, chi, kappa, delta_0, rho_i, phi_pi, phi_y, psi, rho_A, rho_Z, rho_G, rho_nu, rho_psi, s_A, s_Z, s_G, s_nu, s_psi, s_i, s_w, omega, G_gov, F_prod, i_ss, i_LB, division_eps


def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}
    
    weight  = 1.0
    weight2 = 1.0
   
    Ky = Definitions.Ky(state, policy_state)
    iy = Definitions.iy(state, policy_state)
    
    lambday_norm = Definitions.lambday_norm(state, policy_state)
    Cy_norm      = Definitions.Cy_norm(state, policy_state)
    piy_norm     = Definitions.piy_norm(state, policy_state)
    pihashy_norm = Definitions.pihashy_norm(state, policy_state)
    muy_norm     = Definitions.muy_norm(state, policy_state)
    Iy_norm      = Definitions.Iy_norm(state, policy_state)
    Ry_norm      = Definitions.Ry_norm(state, policy_state)
    h1y_norm     = Definitions.h1y_norm(state, policy_state)
    h2y_norm     = Definitions.h2y_norm(state, policy_state)
    whashy_norm  = Definitions.whashy_norm(state, policy_state)
    wy_norm      = Definitions.wy_norm(state, policy_state)
    Ny_norm      = Definitions.Ny_norm(state, policy_state)
    mcy_norm     = Definitions.mcy_norm(state, policy_state)
    x1y_norm     = Definitions.x1y_norm(state, policy_state)
    x2y_norm     = Definitions.x2y_norm(state, policy_state)
    Yy_norm      = Definitions.Yy_norm(state, policy_state)
    nupy_norm    = Definitions.nupy_norm(state, policy_state)

    #loss_dict['eq_1'] = 1600*E_t(lambda s, ps: Cy_norm * Definitions.Cy_norm(s, ps) * lambday_norm - Definitions.Cy_norm(s, ps) * State.nux(state) + Cy_norm * beta * b_habit * State.nux (s)) 
    
  # original eq 1
    loss_dict['eq_1'] = lambday_norm - (State.nux(state) / (Cy_norm)) + beta * b_habit * E_t(lambda s, ps: State.nux (s) / (Definitions.Cy_norm(s, ps)))      
    
    
    loss_dict['eq_2'] = lambday_norm - beta * (1.0 + iy) * E_t(lambda s, ps: Definitions.lambday_norm(s, ps)*(1.0/(1.0 + Definitions.piy_norm(s, ps))))

    loss_dict['eq_3'] = lambday_norm - muy_norm * State.Zx(state) * ((1.0 - kappa/2.0 * (Iy_norm/State.Ix(state) - 1.0 )**2.0) - kappa * (Iy_norm/State.Ix(state) - 1.0) * Iy_norm/State.Ix(state)) - beta * E_t(lambda s, ps: Definitions.muy_norm(s, ps) * State.Zx(s) * kappa * (Definitions.Iy_norm(s, ps)/Iy_norm - 1.0) * (Definitions.Iy_norm(s, ps)/Iy_norm)**2.0) 

    loss_dict['eq_4'] = muy_norm - beta * E_t(lambda s, ps: Definitions.lambday_norm(s, ps) * Definitions.Ry_norm(s, ps) + Definitions.muy_norm(s, ps) * (1.0 - delta_0))

    loss_dict['eq_5'] = h1y_norm * whashy_norm**(eps_w * (1.0 + chi)) - State.nux(state) * State.psix(state) * wy_norm**(eps_w * (1.0 + chi)) * Ny_norm**(1.0 + chi) - phi_w * beta * (1.0 + piy_norm)**(-zeta_w * eps_w * (1.0 + chi )) * E_t(lambda s, ps: (1.0 + Definitions.piy_norm(s, ps))**(eps_w * (1.0 + chi)) * Definitions.whashy_norm(s, ps)**(eps_w * (1.0 + chi)) * Definitions.h1y_norm(s, ps))    
    
    loss_dict['eq_6'] = h2y_norm * whashy_norm**eps_w - lambday_norm * wy_norm**eps_w * Ny_norm - phi_w * beta * (1.0 - piy_norm)**(zeta_w * (1.0 - eps_w)) * E_t(lambda s, ps: (1.0 + Definitions.piy_norm(s, ps))**(eps_w -1.0 ) * Definitions.whashy_norm(s, ps)**eps_w * Definitions.h2y_norm(s, ps))
    
    loss_dict['eq_7'] = whashy_norm * h2y_norm - ( eps_w / (eps_w - 1.0)) * h1y_norm 
    
    loss_dict['eq_8'] = wy_norm * Ny_norm - ((1.0 - alpha) / alpha) * Ky * Ry_norm
        
    loss_dict['eq_9'] = (1.0 - alpha) * State.Ax(state) * mcy_norm * Ky**alpha - wy_norm * Ny_norm**alpha    

    loss_dict['eq_10'] = x1y_norm - lambday_norm * mcy_norm * Yy_norm - phi_p * beta * (1.0 + piy_norm)**(-zeta_p * eps_p) * E_t(lambda s, ps: (1.0 + Definitions.piy_norm(s, ps))**eps_p * Definitions.x1y_norm(s, ps))
    
    loss_dict['eq_11'] = x2y_norm - lambday_norm * Yy_norm - phi_p * beta * ( 1.0 + piy_norm)**(zeta_p *(1.0 - eps_p)) * E_t(lambda s, ps: (1.0 + Definitions.piy_norm(s, ps))**(eps_p - 1.0) * Definitions.x2y_norm(s, ps))    
    
    loss_dict['eq_12'] = (1.0 + pihashy_norm) * x2y_norm - (eps_p / (eps_p - 1.0)) * (1.0 + piy_norm)* x1y_norm    
    
    loss_dict['eq_13'] = Yy_norm - Cy_norm - b_habit * State.Cx(state) - Iy_norm - State.Gx(state) 
    
    loss_dict['eq_14'] = State.Ax(state) * Ky**alpha * Ny_norm**(1.0 - alpha) - F_prod - Yy_norm * nupy_norm  
    
    loss_dict['eq_15'] = nupy_norm * (1.0 + piy_norm)**(- eps_p) - (1.0 - phi_p) * (1.0 + pihashy_norm)**(- eps_p) - (1.0 + State.pix(state))**(-zeta_p*eps_p) * phi_p * State.nupx(state)    
    
    loss_dict['eq_16'] =  (1.0 + piy_norm)**(1.0 - eps_p) - (1.0 - phi_p) * (1.0 + pihashy_norm)**(1.0 - eps_p) - (1.0 + State.pix(state))**(zeta_p * (1.0 - eps_p)) * phi_p 
    
    loss_dict['eq_17'] = wy_norm**(1.0 - eps_w) - (1.0 - phi_w)*whashy_norm**(1.0 - eps_w) - (1.0 + State.pix(state))**(zeta_w * (1.0 - eps_w)) * phi_w * (1.0 + piy_norm)**(eps_w - 1.0) * State.wx(state)**(1.0 - eps_w)
    
    return loss_dict
