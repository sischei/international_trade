# -*- coding: utf-8 -*-
#from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n, N_coutries

N_coutries = 2

#endogenous states
end_state = [{'name': 'Kx'}
         ]

#comment that out later
N = N_coutries
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
print(state)

################################################
# Policies

AD_n_policy = []
for i in range(1,N+1):
    AD_n_policy.append({'name':'AD_'+str(i) + '_y'})

YS_n_policy = []
for i in range(1,N+1):
    YS_n_policy.append({'name':'YS_'+str(i) + '_y'})

YD_n_policy = []
for i in range(1,N+1):
    YD_n_policy.append({'name':'YD_'+str(i) + '_y'})

Y_n_policy = []
for i in range(1,N+1):
    Y_n_policy.append({'name':'Y_'+str(i) + '_y'})

w_n_policy = []
for i in range(1,N+1):
    w_n_policy.append({'name':'w_'+str(i) + '_y'})
    
r_n_policy = []
for i in range(1,N+1):
    r_n_policy.append({'name':'r_'+str(i) + '_y'})    

b_n_policy = []
for i in range(1,N+1):
    b_n_policy.append({'name':'b_'+str(i) + '_y'}) 
    
pD_n_policy = []
for i in range(1,N+1):
    pD_n_policy.append({'name':'pD_'+str(i) + '_y'})    
    
piD_n_i_policy = []
for i in range(1,N+1):
    for j in range(1,N+1):
        piD_n_i_policy.append({'name':'piD_'+str(i) + str(j) + '_y'})    

K_n_policy = []
for i in range(1,N+1):
    K_n_policy.append({'name':'K_'+str(i) + '_y'})   
    
XD_n_policy = []
for i in range(1,N+1):
    XD_n_policy.append({'name':'XD_'+str(i) + '_y'})     

policies = AD_n_policy + YS_n_policy + YD_n_policy + Y_n_policy + w_n_policy + r_n_policy + b_n_policy + pD_n_policy + piD_n_i_policy + K_n_policy + XD_n_policy
print(policies)


#state = [{'name': 'TFP'},
  #{'name': 'depr'},
  #{'name': 'K_total'}]
#middle = [
  #{'name': 'r'},
  #{'name': 'w'},
  #{'name': 'Y'}]
#N = 10
#for i in range(2,N+1):
    #state.append({'name':'K'+str(i)})
#state = state + middle
#for i in range(2,N+1):
    #state.append({'name':'fw'+str(i)})
#print(state)



states = [{'name': 'TFP'},
  {'name': 'depr'},
  {'name': 'K2'},
  {'name': 'K3'},
  {'name': 'K4'},
  {'name': 'K5'},
  {'name': 'K6'},
  {'name': 'K_total'},
  {'name': 'r'},
  {'name': 'w'},
  {'name': 'Y'},
  {'name': 'fw2'},
  {'name': 'fw3'},
  {'name': 'fw4'},
  {'name': 'fw5'},
  {'name': 'fw6'}]

policies = [{'name': 'a1', 'bounds': {'lower': 1e-5}},
  {'name': 'a2', 'bounds': {'lower': 1e-5}},
  {'name': 'a3', 'bounds': {'lower': 1e-5}},
  {'name': 'a4', 'bounds': {'lower': 1e-5}},
  {'name': 'a5', 'bounds': {'lower': 1e-5}}]
 
definitions = [{'name': 'c1', 'bounds': {'lower': 1e-4}},
  {'name': 'c2', 'bounds': {'lower': 1e-4}},
  {'name': 'c3', 'bounds': {'lower': 1e-4}},
  {'name': 'c4', 'bounds': {'lower': 1e-4}},
  {'name': 'c5', 'bounds': {'lower': 1e-4}},
  {'name': 'c6', 'bounds': {'lower': 1e-4}},
  {'name': 'K_total', 'bounds': {'lower': 1e-4}},
  {'name': 'K_total_next', 'bounds': {'lower': 1e-4}},
  {'name': 'r'},
  {'name': 'w'},
  {'name': 'Y'}]
