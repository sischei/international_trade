import tensorflow as tf
import math
import itertools
import Definitions
import State
import Parameters
from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n, N_coutries, sigma_Tn, sigma_phi_n, sigma_chi_n, sigma_d_n_i, rho_dni

import PolicyState

global POST_PROCESSING
POST_PROCESSING=True


##############################################################################
# shocks for monomial integration
# N shocks 
sigma_T_n = []
for i in range(1,N_coutries + 1):
    sigma_T_n.append(sigma_Tn) 

# N-1 shocks #TBD -> FIX correct range
sigmaPhi_n = []
for i in range(1,N_coutries + 1):
    sigmaPhi_n.append(sigma_phi_n) 

# N shocks 
sigmaChi_n = []
for i in range(1,N_coutries + 1):
    sigmaChi_n.append(sigma_chi_n) 

# N x (N-1) shocks in principle. Diagonal elements will be set to zero.
sigma_d_n_i_state = []
for i in range(1,N+1):
    for j in range(1,N+1):
        sigma_d_n_i_state.append(sigma_d_n_i)

##############################################################################


# shock values for monomial integration
vola_tot = sigma_T_n + sigmaPhi_n + sigmaChi_n + sigma_d_n_i_state
print("size of vola shock vec", len(vola_tot))

if Parameters.expectation_type == 'monomial':
    shock_values, shock_probs = State.monomial_rule([vola_tot])


##############################################################################


# update states and policies
def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    shock = shock_step_random(prev_state)
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy        
    return augment_state(total)

# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(prev_state, policy_state, shock_index):
    ar = AR_step(prev_state)
    shock = shock_step_spec_shock(prev_state, shock_index)
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy        
    return augment_state(total)

def augment_state(state):
    for i in range(len(sigma_T_n)):
        state = State.update(state, 'TD_'+str(i) + '_x', tf.math.exp(State.__dict__[ 'TD_'+str(i) + '_x'](state)))
        
    for i in range(len(sigmaPhi_n)):
        state = State.update(state, 'phi_'+str(i) + '_x', tf.math.exp(State.__dict__[ 'phi_'+str(i) + '_x'](state)))    
        
    for i in range(len(sigmaChi_n)):
        state = State.update(state, 'chi_'+str(i) + '_x', tf.math.exp(State.__dict__[ 'chi_'+str(i) + '_x'](state)))        
        
    for i in range(1,N+1):
        for j in range(1,N+1):
            sigma_d_n_i_state.append(sigma_d_n_i) 
            state = State.update(state, 'd_'+str(i) + str(j) + '_x', tf.math.exp(State.__dict__[ 'd_'+str(i) +str(i) + str(j) + '_x'](state)))  
            
    return state    
    
    
    
def AR_step(prev_state):
    # states with autoregressive components    
    ar_step = tf.zeros_like(prev_state)    
    
    for i in range(len(sigma_T_n)):
        ar_step = State.update(ar_step,'TD_'+str(i) + '_x', rhoT_n*tf.math.log(State.__dict__[ 'TD_'+str(i) + '_x'](state)))
        
    for i in range(len(sigmaPhi_n)):
        ar_step = State.update(ar_step,'phi_'+str(i) + '_x', phi_n*tf.math.log(State.__dict__[ 'phi_'+str(i) + '_x'](state)))        
        
    for i in range(len(sigmaChi_n)):
        ar_step = State.update(ar_step,'chi_'+str(i) + '_x', rhoChi_n*tf.math.log(State.__dict__[ 'chi_'+str(i) + '_x'](state)))        
        
    for i in range(1,N+1):
        for j in range(1,N+1):
            ar_step = State.update(ar_step, 'd_'+str(i) + str(j) + '_x', (1.0 - rho_dni)*bar_dni + rho_dni.math.log(State.__dict__[  'd_'+str(i) + str(j) + '_x'](state)))  
            if (i == j):
                ar_step = State.update(ar_step, 'd_'+str(i) + str(j) + '_x', 0.0)  

    return ar_step

    
    
def shock_step_random(prev_state):
    shock_step = tf.zeros_like(prev_state)
    # number of shocks over which we have to integrate - check carefully once code runs #TBD -> FIX correct range
    int_size = N_coutries*N_coutries*N_coutries*(N_coutries*N_coutries)
    random_normals = Parameters.rng.normal([prev_state.shape[0],int_size])    
    
    counter = 0
    for i in range(len(sigma_T_n)):
        shock_step = State.update(shock_step,'TD_'+str(i) + '_x', random_normals[:,counter]*sigma_Tn)
        counter = counter + 1
    
    for i in range(len(sigmaPhi_n)):
        shock_step = State.update(shock_step,'phi_'+str(i) + '_x', random_normals[:, counter]*sigma_phi_n)
        counter = counter + 1
    
    for i in range(len(sigmaChi_n)):
        shock_step = State.update(shock_step,'chi_'+str(i) + '_x', random_normals[:,counter]*sigma_chi_n)
        counter = counter + 1
 
    for i in range(1,N+1):
        for j in range(1,N+1):
            shock_step = State.update(shock_step,'d_'+str(i) + str(j) + '_x', random_normals[:, counter]*sigma_d_n_i)
            counter = counter + 1
    
    return shock_step
    
 

def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(prev_state)
        
    counter = 0
    for i in range(len(sigma_T_n)):
        shock_step = State.update(shock_step,'TD_'+str(i) + '_x', tf.repeat(shock_values[shock_index,counter], prev_state.shape[0])))
        counter = counter + 1    
    
    for i in range(len(sigmaPhi_n)):
        shock_step = State.update(shock_step,'phi_'+str(i) + '_x', tf.repeat(shock_values[shock_index, counter], prev_state.shape[0]))
        counter = counter + 1
    
    for i in range(len(sigmaChi_n)):
        shock_step = State.update(shock_step,'chi_'+str(i) +  '_x', tf.repeat(shock_values[shock_index,counter], prev_state.shape[0]))
        counter = counter + 1
 
    for i in range(1,N+1):
        for j in range(1,N+1):
            shock_step = State.update(shock_step,'d_'+str(i) + str(j) + '_x',tf.repeat(shock_values[shock_index,counter], prev_state.shape[0]))
            counter = counter + 1    
  
    return shock_step


if not POST_PROCESSING:
    print("CUBE DYNAMICS INITIALIZED")
    # cube samplilng dynamics
    def policy_step(prev_state, policy_state):
    # sample from ranges instead of doing a proper dynamics update
        policy_step = tf.zeros_like(prev_state)
        for i in range(len(sigma_T_n)):
            policy_step = State.update(policy_step,'K_'+str(i) + '_x', (State.__dict__[ 'K_'+str(i) + '_x'](prev_state)))
        return policy_state


##original dynamics
if POST_PROCESSING:
    print("SIMULATION DYNAMICS INITIALIZED")    
    def policy_step(prev_state, policy_state):
        # coming from the lagged policy / definition
        policy_step = tf.zeros_like(prev_state)
        
        for i in range(1,N+1):
            policy_step = State.update(policy_step,'K_'+str(i) + '_x', (PolicyState.__dict__[ 'K_'+str(i) + '_y'](policy_state)))
        
        return policy_state
