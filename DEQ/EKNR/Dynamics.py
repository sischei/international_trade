import tensorflow as tf
import math
import itertools
import Definitions
import State
import Parameters
from Parameters import theta, sigma, omega_n, betaL , delta, alpha, rho, rhoT_n, phi_n, rhoChi_n, bar_dni, d_nn, L_n
import PolicyState

global POST_PROCESSING
POST_PROCESSING=True

# shocks
shocks_Ax = [x * math.sqrt(2.0) * s_A for x in [-1.224744871, 0.0, +1.224744871]]
probs_Ax = [x / math.sqrt(math.pi) for x in [0.2954089751, 1.181635900, 0.2954089751]]

shocks_Gx = [x * math.sqrt(2.0) * s_G for x in [-1.224744871, 0.0, +1.224744871]]
probs_Gx = probs_Ax
   
shocks_Zx = [x * math.sqrt(2.0) * s_Z for x in [-1.224744871, 0.0, +1.224744871]]
probs_Zx = probs_Ax

shocks_nux = [x * math.sqrt(2.0) * s_nu for x in [-1.224744871, 0.0, +1.224744871]]
probs_nux = probs_Ax

shocks_psix = [x * math.sqrt(2.0) * s_psi for x in [-1.224744871, 0.0, +1.224744871]]
probs_psix = probs_Ax

shocks_mx = [x * math.sqrt(2.0) * s_i for x in [-1.224744871, 0.0, +1.224744871]]
probs_mx = probs_Ax

shock_values = tf.constant(list(itertools.product(shocks_Ax, shocks_Gx, shocks_Zx, shocks_nux, shocks_psix, shocks_mx)))
shock_probs = tf.constant([ p_ax * p_gx * p_zx * p_nux * p_psix * p_mx for p_ax, p_gx, p_zx, p_nux, p_psix, p_mx in list(itertools.product(probs_Ax, probs_Gx, probs_Zx, probs_nux, probs_psix, probs_mx))])

if Parameters.expectation_type == 'monomial':
    shock_values, shock_probs = State.monomial_rule([s_A, s_G, s_Z, s_nu, s_psi, s_i])

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
    state = State.update(state, "Ax", tf.math.exp(State.Ax(state)))
    state = State.update(state, "Gx", tf.math.exp(State.Gx(state)))
    state = State.update(state, "Zx", tf.math.exp(State.Zx(state)))
    state = State.update(state, "nux", tf.math.exp(State.nux(state)))
    return State.update(state, "psix", tf.math.exp(State.psix(state)))

def AR_step(prev_state):
    # states with autoregressive components    
    ar_step = tf.zeros_like(prev_state)
    ar_step = State.update(ar_step, "Ax", Parameters.rho_A * tf.math.log(State.Ax(prev_state)))
    ar_step = State.update(ar_step, "Gx", (1.0 - Parameters.rho_G ) * tf.math.log(Parameters.G_gov) + Parameters.rho_G * tf.math.log(State.Gx(prev_state)))
    ar_step = State.update(ar_step, "Zx", Parameters.rho_Z * tf.math.log(State.Zx(prev_state)))
    ar_step = State.update(ar_step, "nux", Parameters.rho_nu * tf.math.log(State.nux(prev_state)))
    return State.update(ar_step, "psix", (1.0 - Parameters.rho_psi)*tf.math.log(Parameters.psi) + Parameters.rho_psi*tf.math.log(State.psix(prev_state)))

def shock_step_random(prev_state):
    shock_step = tf.zeros_like(prev_state)
    random_normals = Parameters.rng.normal([prev_state.shape[0],3])
    shock_step = State.update(shock_step, "rf", random_normals[:,0] * Parameters.sigma_rf)
    shock_step = State.update(shock_step, "yT", random_normals[:,1] * Parameters.sigma_yT)
    return State.update(shock_step, "delta", random_normals[:,2] * Parameters.sigma_delta)

def shock_step_random(prev_state): 
    shock_step = tf.zeros_like(prev_state)
    random_normals = Parameters.rng.normal([prev_state.shape[0],6])
    shock_step = State.update(shock_step, "Ax", random_normals[:,0] * Parameters.s_A)
    shock_step = State.update(shock_step, "Gx", random_normals[:,1] * Parameters.s_G)
    shock_step = State.update(shock_step, "Zx", random_normals[:,2] * Parameters.s_Z)
    shock_step = State.update(shock_step, "nux", random_normals[:,3] * Parameters.s_nu)   
    shock_step = State.update(shock_step, "psix", random_normals[:,4] * Parameters.s_psi)
    return State.update(shock_step, "mx", random_normals[:,5] * Parameters.s_i)

def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(prev_state)
    shock_step = State.update(shock_step,"Ax", tf.repeat(shock_values[shock_index,0], prev_state.shape[0]))
    shock_step = State.update(shock_step,"Gx", tf.repeat(shock_values[shock_index,1], prev_state.shape[0]))
    shock_step = State.update(shock_step,"Zx", tf.repeat(shock_values[shock_index,2], prev_state.shape[0]))
    shock_step = State.update(shock_step,"nux", tf.repeat(shock_values[shock_index,3], prev_state.shape[0]))
    shock_step = State.update(shock_step,"psix", tf.repeat(shock_values[shock_index,4], prev_state.shape[0]))
    return State.update(shock_step,"mx", tf.repeat(shock_values[shock_index,5], prev_state.shape[0]))


if not POST_PROCESSING:
    print("CUBE DYNAMICS INITIALIZED")
    # cube samplilng dynamics
    def policy_step(prev_state, policy_state):
    # sample from ranges instead of doing a proper dynamics update
        policy_step = tf.zeros_like(prev_state)
        policy_step = State.update(policy_step, "Kx",State.Kx(prev_state))
        policy_step = State.update(policy_step, "Cx",State.Cx(prev_state))
        policy_step = State.update(policy_step, "Ix",State.Ix(prev_state))   
        policy_step = State.update(policy_step, "Yx",State.Yx(prev_state))
        policy_step = State.update(policy_step, "wx",State.wx(prev_state))
        policy_step = State.update(policy_step, "nupx",State.nupx(prev_state))
        policy_step = State.update(policy_step, "pix", State.pix(prev_state))    
        return State.update(policy_step,"ix", State.ix(prev_state))


##original dynamics
if POST_PROCESSING:
    print("SIMULATION DYNAMICS INITIALIZED")    
    def policy_step(prev_state, policy_state):
        # coming from the lagged policy / definition
        policy_step = tf.zeros_like(prev_state)
        #policy_step = State.update(policy_step, "Kx",PolicyState.Ky(policy_state))
        policy_step = State.update(policy_step, "Kx",Definitions.Ky(prev_state, policy_state)) 
        policy_step = State.update(policy_step, "Cx", (Definitions.Cy_norm(prev_state, policy_state) + Parameters.b_habit * State.Cx(prev_state)))
        policy_step = State.update(policy_step, "Ix", (Definitions.Iy_norm(prev_state, policy_state)))
        policy_step = State.update(policy_step, "Yx",Definitions.Yy_norm(prev_state, policy_state))
        policy_step = State.update(policy_step, "wx",Definitions.wy_norm(prev_state, policy_state))
        policy_step = State.update(policy_step, "nupx",Definitions.nupy_norm(prev_state, policy_state))
        policy_step = State.update(policy_step, "pix", Definitions.piy_norm(prev_state, policy_state))
        return State.update(policy_step,"ix", Definitions.iy(prev_state, policy_state))
    
