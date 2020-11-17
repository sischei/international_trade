import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import alpha, beta, b_habit, eps_p, eps_w, phi_w, phi_p, pi_ss, zeta_w, zeta_p, chi, kappa, delta_0, delta_2, rho_i, phi_pi, phi_y, psi, rho_A, rho_Z, rho_G, rho_nu, rho_psi, s_A, s_Z, s_G, s_nu, s_psi, s_i, s_w, omega, G_gov, F_prod, i_ss, i_LB, division_eps


def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}

    delta_1 = 1.0/beta - (1.0 - delta_0)
    #delta_1 = Definitions.delta_1(state, policy_state)
    
    weight = 0.6
    
   
    #original equation
    #loss_dict['eq_1'] = tf.exp(E_t(lambda s, ps: PolicyState.Cy(policy_state) * PolicyState.Cy(ps) * PolicyState.lambday(policy_state) - PolicyState.Cy(ps) * State.nux(state) + PolicyState.Cy(policy_state) * beta * b_habit * State.nux (s))) - 1.0    
    
    # original eq 1
    #loss_dict['eq_1'] = PolicyState.lambday(policy_state) - (State.nux(state) / (PolicyState.Cy(policy_state) )) + beta * b_habit * E_t(lambda s, ps: State.nux (s) / (PolicyState.Cy(ps)))    
    
    
    
      # original eq 1 1/C
    #loss_dict['eq_1'] = PolicyState.lambday(policy_state) - (State.nux(state) / (PolicyState.Cy(policy_state) )) + beta * b_habit * E_t(lambda s, ps: State.nux (s) / ((weight*(PolicyState.Cy(ps)) + (1-weight)*0.025)))    
    
    
    # debug #  replace (PolicyState.Cy(ps)) with (weight*(PolicyState.Cy(ps)) + (1-weight)*0.025) -- E()
    loss_dict['eq_1'] = E_t(lambda s, ps: PolicyState.Cy(policy_state) * (weight*(PolicyState.Cy(ps)) + (1-weight)*0.025) * PolicyState.lambday(policy_state) - (weight*(PolicyState.Cy(ps)) + (1-weight)*0.025) * State.nux(state) + PolicyState.Cy(policy_state) * beta * b_habit * State.nux (s)) 
    
    loss_dict['eq_2'] = PolicyState.lambday(policy_state) * PolicyState.Ry(policy_state) - PolicyState.muy(policy_state) * (delta_1 + delta_2 *(PolicyState.uy(policy_state) - 1.0)) 
    
    #loss_dict['eq_3'] = PolicyState.lambday(policy_state) - beta * (1.0 + PolicyState.iy(policy_state)) * E_t(lambda s, ps: PolicyState.lambday(ps)*(1.0/(1.0 + PolicyState.piy(ps) )))        
    
    # debug: replace PolicyState.piy(ps) with (weight*(PolicyState.piy(ps)) + (1-weight)*0.005)        
    # debug: replace PolicyState.lambday(ps) with (weight*(PolicyState.lambday(ps)) + (1-weight)*2.6206) 
    loss_dict['eq_3'] = PolicyState.lambday(policy_state) - beta * (1.0 + PolicyState.iy(policy_state)) * E_t(lambda s, ps: PolicyState.lambday(ps)*(1.0/(1.0 + (weight*(PolicyState.piy(ps)) + (1-weight)*0.005))))
    
    #loss_dict['eq_4'] = PolicyState.lambday(policy_state) - PolicyState.muy(policy_state) * State.Zx(state) * ((1.0 - kappa/2.0 * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0 )**2.0) - kappa * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0) * PolicyState.Iy(policy_state)/State.Ix(state)) - beta * E_t(lambda s, ps: PolicyState.muy(ps) * State.Zx(s) * kappa * (PolicyState.Iy(ps)/PolicyState.Iy(policy_state) - 1.0) * (PolicyState.Iy(ps)/PolicyState.Iy(policy_state))**2.0 )
    
    # debug: replace PolicyState.Iy(ps) with (weight*(PolicyState.Iy(ps)) + (1-weight)*0.1881)  
    # debug: replace PolicyState.muy(ps) with (weight*(PolicyState.muy(ps)) + (1-weight)*2.6206) 
    loss_dict['eq_4'] = PolicyState.lambday(policy_state) - PolicyState.muy(policy_state) * State.Zx(state) * ((1.0 - kappa/2.0 * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0 )**2.0) - kappa * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0) * PolicyState.Iy(policy_state)/State.Ix(state)) - beta * E_t(lambda s, ps: (weight*(PolicyState.muy(ps)) + (1-weight)*2.6206) * State.Zx(s) * kappa * ((weight*(PolicyState.Iy(ps)) + (1-weight)*0.1881)/PolicyState.Iy(policy_state) - 1.0) * ((weight*(PolicyState.Iy(ps)) + (1-weight)*0.1881)/PolicyState.Iy(policy_state))**2.0 )
    
    #loss_dict['eq_5'] = PolicyState.muy(policy_state) - beta * E_t(lambda s, ps: PolicyState.lambday(ps) * PolicyState.Ry(ps) * PolicyState.uy(ps) + PolicyState.muy(ps) * (1.0 - (delta_0 + delta_1 * (PolicyState.uy(ps) - 1.0) + delta_2/2.0 * (PolicyState.uy(ps) - 1.0)**2.0 )))   

    # debug: replace PolicyState.muy(ps) with (weight*(PolicyState.muy(ps)) + (1-weight)*2.6206) 
    # debug: replace PolicyState.uy(ps) with (weight*(PolicyState.uy(ps)) + (1-weight)*1.0) 
    # debug: replace PolicyState.lambday(ps) with (weight*(PolicyState.lambday(ps)) + (1-weight)*2.6206)     
    loss_dict['eq_5'] = PolicyState.muy(policy_state) - beta * E_t(lambda s, ps: (weight*(PolicyState.lambday(ps)) + (1-weight)*2.6206) * PolicyState.Ry(ps) * PolicyState.uy(ps) + (weight*(PolicyState.muy(ps)) + (1-weight)*2.6206) * (1.0 - (delta_0 + delta_1 * ((weight*(PolicyState.uy(ps)) + (1-weight)*1.0) - 1.0) + delta_2/2.0 * ((weight*(PolicyState.uy(ps)) + (1-weight)*1.0) - 1.0)**2.0 )))  
    
    #loss_dict['eq_6'] = PolicyState.h1y(policy_state) * PolicyState.whashy(policy_state)**(eps_w * (1.0 + chi)) - State.nux(state) * State.psix(state) * PolicyState.wy(policy_state)**(eps_w * (1.0 + chi)) * PolicyState.Ny(policy_state)**(1.0 + chi) - phi_w * beta * (1.0 + PolicyState.piy(policy_state))**(-zeta_w * eps_w * ( 1.0 + chi )) * E_t(lambda s, ps: (1.0 + PolicyState.piy(ps))**(eps_w * (1.0 + chi)) * PolicyState.whashy(ps)**(eps_w * (1.0 + chi)) * PolicyState.h1y(ps))
 
    # debug: replace PolicyState.whashy(ps) with (weight*(PolicyState.whashy(ps)) + (1-weight)*1.969) 
    # debug: replace PolicyState.h1y(ps) with (weight*(PolicyState.h1y(ps)) + (1-weight)*1.6711)     
    # debug: replace PolicyState.piy(ps) with (weight*(PolicyState.piy(ps)) + (1-weight)*0.005)         
    loss_dict['eq_6'] = PolicyState.h1y(policy_state) * PolicyState.whashy(policy_state)**(eps_w * (1.0 + chi)) - State.nux(state) * State.psix(state) * PolicyState.wy(policy_state)**(eps_w * (1.0 + chi)) * PolicyState.Ny(policy_state)**(1.0 + chi) - phi_w * beta * (1.0 + PolicyState.piy(policy_state))**(-zeta_w * eps_w * ( 1.0 + chi )) * E_t(lambda s, ps: (1.0 + (weight*(PolicyState.piy(ps)) + (1-weight)*0.005))**(eps_w * (1.0 + chi)) * (weight*(PolicyState.whashy(ps)) + (1-weight)*1.969)**(eps_w * (1.0 + chi)) * (weight*(PolicyState.h1y(ps)) + (1-weight)*1.6711))
 
    #loss_dict['eq_7'] = PolicyState.h2y(policy_state) * PolicyState.whashy(policy_state)**eps_w - PolicyState.lambday(policy_state) * PolicyState.wy(policy_state)**eps_w * PolicyState.Ny(policy_state) - phi_w * beta * (1.0 - PolicyState.piy(policy_state))**(zeta_w * (1.0 - eps_w)) * E_t(lambda s, ps: (1.0 + PolicyState.piy(ps))**(eps_w -1.0 ) * PolicyState.whashy(ps)**eps_w * PolicyState.h2y(ps))
    
    # debug: replace PolicyState.whashy(ps) with (weight*(PolicyState.whashy(ps)) + (1-weight)*1.969) 
    # debug: replace PolicyState.h2y(ps) with (weight*(PolicyState.h2y(ps)) + (1-weight)*2.8441)         
    # debug: replace PolicyState.piy(ps) with (weight*(PolicyState.piy(ps)) + (1-weight)*0.005)             
    loss_dict['eq_7'] = PolicyState.h2y(policy_state) * PolicyState.whashy(policy_state)**eps_w - PolicyState.lambday(policy_state) * PolicyState.wy(policy_state)**eps_w * PolicyState.Ny(policy_state) - phi_w * beta * (1.0 - PolicyState.piy(policy_state))**(zeta_w * (1.0 - eps_w)) * E_t(lambda s, ps: (1.0 + (weight*(PolicyState.piy(ps)) + (1-weight)*0.005))**(eps_w -1.0 ) * (weight*(PolicyState.whashy(ps)) + (1-weight)*1.969)**eps_w * (weight*(PolicyState.h2y(ps)) + (1-weight)*2.8441))
    
    loss_dict['eq_8'] = PolicyState.whashy(policy_state) * PolicyState.h2y(policy_state) - ( eps_w / (eps_w - 1.0)) * PolicyState.h1y(policy_state)
    
    loss_dict['eq_9'] = PolicyState.wy(policy_state) *  PolicyState.Ny(policy_state) - ((1.0 - alpha) / alpha) * PolicyState.Khaty(policy_state) *  PolicyState.Ry(policy_state)
    
    loss_dict['eq_10'] = (1.0 - alpha) * State.Ax(state) * PolicyState.mcy(policy_state) * PolicyState.Khaty(policy_state)**alpha - PolicyState.wy(policy_state) * PolicyState.Ny(policy_state)**alpha
        
    #loss_dict['eq_11'] = PolicyState.x1y(policy_state) - PolicyState.lambday(policy_state) * PolicyState.mcy(policy_state) * PolicyState.Yy(policy_state) - phi_p * beta * (1.0 + PolicyState.piy(policy_state))**(-zeta_p * eps_p) * E_t(lambda s, ps: (1.0 + PolicyState.piy(ps))**eps_p * PolicyState.x1y(ps) )

    # debug: replace PolicyState.x1y(ps) with (weight*(PolicyState.x1y(ps)) + (1-weight)*5.3470) 
    # debug: replace PolicyState.piy(ps) with (weight*(PolicyState.piy(ps)) + (1-weight)*0.005)                 
    loss_dict['eq_11'] = PolicyState.x1y(policy_state) - PolicyState.lambday(policy_state) * PolicyState.mcy(policy_state) * PolicyState.Yy(policy_state) - phi_p * beta * (1.0 + PolicyState.piy(policy_state))**(-zeta_p * eps_p) * E_t(lambda s, ps: (1.0 + (weight*(PolicyState.piy(ps)) + (1-weight)*0.005))**eps_p * (weight*(PolicyState.x1y(ps)) + (1-weight)*5.3470))

    
    #loss_dict['eq_12'] = PolicyState.x2y(policy_state) - PolicyState.lambday(policy_state) * PolicyState.Yy(policy_state) - phi_p * beta * ( 1.0 + PolicyState.piy(policy_state))**(zeta_p *(1.0 - eps_p)) * E_t(lambda s, ps: (1.0 + PolicyState.piy(ps))**(eps_p - 1.0) * PolicyState.x2y(ps))

    # debug: replace PolicyState.x2y(ps) with (weight*(PolicyState.x2y(ps)) + (1-weight)*6.6009)  
    # debug: replace PolicyState.piy(ps) with (weight*(PolicyState.piy(ps)) + (1-weight)*0.005)                     
    loss_dict['eq_12'] = PolicyState.x2y(policy_state) - PolicyState.lambday(policy_state) * PolicyState.Yy(policy_state) - phi_p * beta * ( 1.0 + PolicyState.piy(policy_state))**(zeta_p *(1.0 - eps_p)) * E_t(lambda s, ps: (1.0 + (weight*(PolicyState.piy(ps)) + (1-weight)*0.005))**(eps_p - 1.0) * (weight*(PolicyState.x2y(ps)) + (1-weight)*6.6009))    
    
    loss_dict['eq_13'] = (1.0 + PolicyState.pihashy(policy_state)) * PolicyState.x2y(policy_state) - (eps_p / (eps_p - 1.0)) * (1.0 + PolicyState.piy(policy_state))* PolicyState.x1y(policy_state)

    loss_dict['eq_14'] = PolicyState.Yy(policy_state) - PolicyState.Cy(policy_state) - b_habit * State.Cx(state) - PolicyState.Iy(policy_state) - State.Gx(state) 
    
    loss_dict['eq_15'] = PolicyState.Ky(policy_state) - State.Zx(state) * (1.0 - kappa/2.0 * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0 )**2.0) * PolicyState.Iy(policy_state) - (1.0 - (delta_0 + delta_1 * (PolicyState.uy(policy_state) - 1.0) + delta_2/2.0 * (PolicyState.uy(policy_state) - 1.0)**2.0)) * State.Kx(state)       
    
    #loss_dict['eq_15'] = tf.square(PolicyState.Ky(policy_state) - State.Zx(state) * (1.0 - kappa/2.0 * (PolicyState.Iy(policy_state)/State.Ix(state) - 1.0 )**2.0) * PolicyState.Iy(policy_state) - (1.0 - (delta_0 + delta_1 * (PolicyState.uy(policy_state) - 1.0) + delta_2/2.0 * (PolicyState.uy(policy_state) - 1.0)**2.0)) * State.Kx(state))    




    loss_dict['eq_16'] = State.Ax(state) * PolicyState.Khaty(policy_state)**alpha * PolicyState.Ny(policy_state)**(1.0 - alpha) - F_prod - PolicyState.Yy(policy_state) * PolicyState.nupy(policy_state)
    
    loss_dict['eq_17'] = PolicyState.Khaty(policy_state) - PolicyState.uy(policy_state) * State.Kx(state)
    
    loss_dict['eq_18'] = PolicyState.nupy(policy_state) * (1.0 + PolicyState.piy(policy_state))**(- eps_p) - (1.0 - phi_p) * (1.0 + PolicyState.pihashy(policy_state))**(- eps_p) - (1.0 + State.pix(state))**(-zeta_p*eps_p) * phi_p * State.nupx(state)
    
    loss_dict['eq_19'] =  (1.0 + PolicyState.piy(policy_state))**(1.0 - eps_p) - (1.0 - phi_p) * (1.0 + PolicyState.pihashy(policy_state) )**(1.0 - eps_p) - (1.0 + State.pix(state))**(zeta_p * (1.0 - eps_p)) * phi_p
    
    loss_dict['eq_20'] = PolicyState.wy(policy_state)**(1.0 - eps_w) - (1.0 - phi_w)*PolicyState.whashy(policy_state)**(1.0 - eps_w) - (1.0 + State.pix(state))**(zeta_w * (1.0 - eps_w)) * phi_w * (1.0 + PolicyState.piy(policy_state))**(eps_w - 1.0) * State.wx(state)**(1.0 - eps_w)

    loss_dict['eq_21'] = PolicyState.iy(policy_state) - tf.math.maximum((1.0 - rho_i) * i_ss + rho_i * State.ix(state) +  (1.0 - rho_i) * (phi_pi * (PolicyState.piy(policy_state) - pi_ss) + phi_y * (tf.math.log(PolicyState.Yy(policy_state)) - tf.math.log(State.Yx(state))) + State.mx(state)), i_LB )

    return loss_dict
