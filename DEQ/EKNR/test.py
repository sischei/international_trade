state = [{'name': 'TFP'},
  {'name': 'depr'},
  {'name': 'K_total'}]
middle = [
  {'name': 'r'},
  {'name': 'w'},
  {'name': 'Y'}]
N = 10
for i in range(2,N+1):
    state.append({'name':'K'+str(i)})
state = state + middle
for i in range(2,N+1):
    state.append({'name':'fw'+str(i)})
print(state)


rho =0.2
rho2 = 0.3
rho4 = 0.5

empty = []
empty = [rho, rho2, rho4]
print(empty)

N = 10
AD_n_policy = []
for i in range(1,N+1):
    rho_1 = 10
    AD_n_policy.append(rho_1)
    
print(AD_n_policy)

empty = empty + AD_n_policy
print(empty)
