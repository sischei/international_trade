import tensorflow as tf

def print_values(self):
        """
        Print all parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)
                
                
class A():
    def __init__(self, foo):
        self.foo = foo

    def new_var(self, bar):
        self.bar = bar
        
a = A('var1')
print(a.__dict__) # {'foo': 'var1'}

b = A('var1')
b.new_var('var2')
b.foobar = 'var3'
print(b.__dict__) # {'foo': 'var1', 'bar': 'var2', 'foobar': 'var3'}


print_values(b)


#def state_wrapper(x):
    ## x is a string that goes into the dicts
    ## phi is the  phi_i,t of equation 15
    #phi= doSomething(State.__dict__[x]) 
    #return phi 
## create a tensor with the list of state to modify
#list_to_mod = tf.convert_to_tensor(['state_'+str(i) for i in I])
## vectorized map is an apply along axis that tensorflow will distribute/paralelize efficiently
#p = tf.vectorized_map(fn=state_wrapper, elems=list_to_mod)
#PHI_N_t = N - tf.reduce_sum(p)
