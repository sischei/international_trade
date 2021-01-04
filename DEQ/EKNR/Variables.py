# -*- coding: utf-8 -*-
#from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n, N_coutries
import tensorflow as tf

#comment that out later
N_coutries = 2
N = N_coutries

#endogenous states
end_state = []
for i in range(1,N+1):
    end_state.append({'name':'K_'+str(i) + '_x'})

#exogenous states
TD_n_state = []
for i in range(1,N+1):
    TD_n_state.append({'name':'TD_'+str(i) + '_x'})

phi_n_state = []
for i in range(1,N+1):
    phi_n_state.append({'name':'phi_'+str(i) + '_x'})

chi_n_state = []
for i in range(1,N+1):
    chi_n_state.append({'name':'chi_'+str(i) + '_x'})

d_n_i_state = []
for i in range(1,N+1):
    for j in range(1,N+1):
        d_n_i_state.append({'name':'d_'+str(i) + str(j) + '_x'})

#total state space
state = end_state + TD_n_state + phi_n_state + chi_n_state + d_n_i_state
#print(state)
print("number of states", len(state)) # 4d + d^2 + 1 states

################################################
# Policies

AD_n_policy = []
for i in range(1,N+1):
    AD_n_policy.append({'name':'AD_'+str(i) + '_y','activation': tf.nn.softplus})

YS_n_policy = []
for i in range(1,N+1):
    YS_n_policy.append({'name':'YS_'+str(i) + '_y','activation': tf.nn.softplus})

YD_n_policy = []
for i in range(1,N+1):
    YD_n_policy.append({'name':'YD_'+str(i) + '_y','activation': tf.nn.softplus})

Y_n_policy = []
for i in range(1,N+1):
    Y_n_policy.append({'name':'Y_'+str(i) + '_y','activation': tf.nn.softplus})

w_n_policy = []
for i in range(1,N+1):
    w_n_policy.append({'name':'w_'+str(i) + '_y','activation': tf.nn.softplus})
    
r_n_policy = []
for i in range(1,N+1):
    r_n_policy.append({'name':'r_'+str(i) + '_y','activation': tf.nn.softplus})    

b_n_policy = []
for i in range(1,N+1):
    b_n_policy.append({'name':'b_'+str(i) + '_y','activation': tf.nn.softplus}) 
    
pD_n_policy = []
for i in range(1,N+1):
    pD_n_policy.append({'name':'pD_'+str(i) + '_y','activation': tf.nn.softplus})    
    
piD_n_i_policy = []
for i in range(1,N+1):
    for j in range(1,N+1):
        piD_n_i_policy.append({'name':'piD_'+str(i) + str(j) + '_y','activation': tf.nn.softplus})    

K_n_policy = []
for i in range(1,N+1):
    K_n_policy.append({'name':'K_'+str(i) + '_y','activation': tf.nn.softplus})   
    
XD_n_policy = []
for i in range(1,N+1):
    XD_n_policy.append({'name':'XD_'+str(i) + '_y','activation': tf.nn.softplus})     

policies = AD_n_policy + YS_n_policy + YD_n_policy + Y_n_policy + w_n_policy + r_n_policy + b_n_policy + pD_n_policy + piD_n_i_policy + K_n_policy + XD_n_policy
#print(policies)
print("number of policies", len(policies)) # 10d + d^2 policies


################################################
# definitions
definitions = [{'name': 'betaK'},
               {'name': 'gamma_param'},
               {'name': 'omega_n'}]
print("number of definitions", definitions)
