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
