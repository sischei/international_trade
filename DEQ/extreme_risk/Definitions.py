import PolicyState
import State
import tensorflow as tf
from scipy.stats import norm
from Parameters import xiw, b_p, mu, sigmaa, Sdoupr_p, xip, rhotil_p, iota_p, iotaw_p, iotamu_p, adytil_p, aptil_p, alpha, beta, bigtheta_p, delta, eta_g, gamma, lambdaf, lambdaw, muzstar_p, pibar, psiL, Re_p, sigmaL, tauc, tauk, taul, upsil, we_p, sigma, phi, rentky_ss, p_12, p_22, rho_theta, s_theta, rho_eps, s_eps, rho_muzstar, s_muzstar, rho_muup, s_muup, kstary_ss, cy_ss, lambdazy_ss, omegabary_ss, ny_ss, wtildey_ss, sy_ss, yzy_ss, hy_ss, Rky_ss, Fpy_ss, Kpy_ss, Fwy_ss, Kwy_ss


def pitilde(state, policy_state):
    return pibar^(iota_p)*State.pix(state)^(1-iota_p)

def pitildew(state, policy_state):
    return pibar^(iotaw_p)*State.pix(state)^(1-iotaw_p)

def piw(state, policy_state):
    return piy_norm(state, policy_state)*State.muzstarx(state)*wtildey_norm(state, policy_state)/State.wtildex(state)

def autil(state, policy_state):
    return rentky_ss * (tf.math.exp(sigmaa * (PolicyState.uy(policy_state) - 1)) - 1) / sigmaa

def s_invest(state, policy_state):
    return tf.math.exp(sqrt(Sdoupr_p / 2)*( State.muzstarx(state) * upsil * PolicyState.iy(policy_state) - muzstar_p * upsil)) + tf.math.exp(-sqrt(Sdoupr_p / 2) * ( State.muzstarx(state) * upsil *  PolicyState.iy(policy_state) - muzstar_p * upsil)) - 2

def spr_invest(state, policy_state):
    return sqrt(Sdoupr_p / 2) * (tf.math.exp(sqrt(Sdoupr_p / 2) * ( State.muzstarx(state) * upsil *PolicyState.iy(policy_state)- muzstar_p * upsil)) - tf.math.exp(-sqrt(Sdoupr_p / 2) * (State.muzstarx(state) * upsil * PolicyState.iy(policy_state) - muzstar_p * upsil)))

def Fomegabar(state, policy_state):
    return norm.cdf(((tf.math.log(omegabary_norm(state, policy_state))+State.dx(state)*State.thetax(state) + sigma^2 / 2) / sigma))

def Gomegabar(state, policy_state):
    return norm.cdf((tf.math.log(omegabary_norm(state, policy_state))+State.dx(state)*State.thetax(state) + sigma^2 / 2) / sigma - sigma))

def kbarx(state, policy_state):
    return State.kstarx(state)*tf.math.exp(-State.dx(state)*State.thetax(state))

def p_d(state, policy_state):
    return (1-State.dx(state))*p_12+State.dx(state)*p_22

### normalized policies
def piy_norm(state, policy_state):
    return PolicyState.piy(policy_state)*pibar

def kstary_norm(state, policy_state):
    return PolicyState.kstary(policy_state)*kstary_ss

def cy_norm(state, policy_state):
    return PolicyState.cy(policy_state)*cy_ss

def lambdazy_norm(state, policy_state):
    return PolicyState.lambdazy(policy_state)*lambdazy_ss

def omegabary_norm(state, policy_state):
    return PolicyState.omegabary(policy_state)*omegabary_ss

def ny_norm(state, policy_state):
    return PolicyState.ny(policy_state)*ny_ss

def Rey_norm(state, policy_state):
    return PolicyState.Rey(policy_state)*(Re_p + 1.0) - 1.0

def wtildey_norm(state, policy_state):
    return PolicyState.wtildey(policy_state)*wtildey_ss

def sy_norm(state, policy_state):
    return PolicyState.sy(policy_state)*sy_ss

def yzy_norm(state, policy_state):
    return PolicyState.yzy(policy_state)*yzy_ss

def hy_norm(state, policy_state):
    return PolicyState.hy(policy_state)*hy_ss

def rentky_norm(state, policy_state):
    return PolicyState.rentky(policy_state)*(rentky_ss + 1.0) - 1.0

def Rky_norm(state, policy_state):
    return PolicyState.Rky(policy_state)*(Rky_ss+1.0) - 1.0

def Fpy_norm(state, policy_state):
    return PolicyState.Fpy(policy_state)*Fpy_ss

def Kpy_norm(state, policy_state):
    return PolicyState.Kpy(policy_state)*Kpy_ss

def Fwy_norm(state, policy_state):
    return PolicyState.Fwy(policy_state)*Fwy_ss

def Kwy_norm(state, policy_state):
    return PolicyState.Kwy(policy_state)*Kwy_ss
