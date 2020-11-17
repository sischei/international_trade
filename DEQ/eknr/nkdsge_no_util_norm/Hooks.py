import Parameters
from Parameters import policy, states, policy_states
import PolicyState
import State
import tensorflow as tf
import Definitions
from Parameters import definitions, definition_bounds_hard, MODEL_NAME

def cycle_hook(state,i):
    policy_state = policy(state)
    for s in states:
        tf.summary.histogram("hist_" + s, getattr(State,s)(state), step=i)
        
    for p in policy_states:
        tf.summary.histogram("hist_" + p, getattr(PolicyState,p)(policy_state), step=i)
        
    for d in definitions:
        tf.summary.histogram("hist_" + d, getattr(Definitions,d)(state, policy_state), step=i)

    return True


## comment out in presence of ranges...
#def post_init():
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Ax",tf.constant(1.0,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Gx",tf.constant(0.15,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Zx",tf.constant(1.0,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "nux",tf.constant(1.0,shape=(Parameters.starting_state.shape[0],))))    
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "psix",tf.constant(6.0,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "mx",tf.constant(0.0,shape=(Parameters.starting_state.shape[0],))))    
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Kx",tf.constant(7.47,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Cx",tf.constant(0.41,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Ix",tf.constant(0.19,shape=(Parameters.starting_state.shape[0],))))    
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "Yx",tf.constant(0.74,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "wx",tf.constant(1.19,shape=(Parameters.starting_state.shape[0],))))    
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "nupx",tf.constant(1.0,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "pix",tf.constant(0.0005,shape=(Parameters.starting_state.shape[0],))))
    #Parameters.starting_state.assign(State.update(Parameters.starting_state, "mx",tf.constant(0.01,shape=(Parameters.starting_state.shape[0],))))     
