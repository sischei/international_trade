import tensorflow as tf
import Definitions
import PolicyState
import State
from scipy.stats import norm
from Parameters import xiw, b_p, mu, sigmaa, Sdoupr_p, xip, rhotil_p, iota_p, iotaw_p, iotamu_p, adytil_p, aptil_p, alpha, beta, bigtheta_p, delta, eta_g, gamma, lambdaf, lambdaw, muzstar_p, pibar, psiL, Re_p, sigmaL, tauc, tauk, taul, upsil, we_p, sigma, phi, rentky_ss, p_12, p_22, rho_theta, s_theta, rho_eps, s_eps, rho_muzstar, s_muzstar, rho_muup, s_muup


def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)

    loss_dict = {}

    weight  = 1.0
    weight2 = 1.0

    # Import definitions
    pitilde = Definitions.pitilde(state, policy_state)
    pitildew = Definitions.pitildew(state, policy_state)
    piw = Definitions.piw(state, policy_state)
    autil = Definitions.autil(state, policy_state)
    s_invest = Definitions.s_invest(state, policy_state)
    spr_invest = Definitions.spr_invest(state, policy_state)
    Fomegabar = Definitions.Fomegabar(state, policy_state)
    Gomegabar = Definitions.Gomegabar(state, policy_state)
    kbarx = Definitions.kbarx(state, policy_state)
    p_d = Definitions.p_d(state, policy_state)
    piy_norm = Definitions.piy_norm(state, policy_state)
    kstary_norm = Definitions.kstary_norm(state, policy_state)
    cy_norm = Definitions.cy_norm(state, policy_state)
    lambdazy_norm = Definitions.lambdazy_norm(state, policy_state)
    omegabary_norm = Definitions.omegabary_norm(state, policy_state)
    ny_norm = Definitions.ny_norm(state, policy_state)
    Rey_norm = Definitions.Rey_norm(state, policy_state)
    wtildey_norm = Definitions.wtildey_norm(state, policy_state)
    sy_norm = Definitions.sy_norm(state, policy_state)
    yzy_norm = Definitions.yzy_norm(state, policy_state)
    hy_norm = Definitions.hy_norm(state, policy_state)
    rentky_norm = Definitions.rentky_norm(state, policy_state)
    Rky_norm = Definitions.Rky_norm(state, policy_state)
    Fpy_norm = Definitions.Fpy_norm(state, policy_state)
    Kpy_norm = Definitions.Kpy_norm(state, policy_state)
    Fwy_norm = Definitions.Fwy_norm(state, policy_state)
    Kwy_norm = Definitions.Kwy_norm(state, policy_state)

###########################################################################################################################
###########################################################################################################################

# EQ 1 : Law of motion for p^*
loss_dict['eq_1'] = PolicyState.pstary(policy_state) - ((1 - xip) * (Kpy_norm / Fpy_norm)**(lambdaf/(1-lambdaf)) + xip * ((pitilde / piy_norm) * State.pstarx(state))**(lambdaf/(1-lambdaf)))**((1-lambdaf)/lambdaf)

# EQ 2 : Law of motion for F_p
loss_dict['eq_2'] =  lambdazy_norm * yzy_norm + E_t(lambda s, ps: (Definitions.piy_norm(s, ps)**iota_p * piy_norm**(1-iota_p) * pibar**(1-iota_p-(1-iota_p)) / Definitions.piy_norm(s, ps))**(1/(1-lambdaf)) * beta * xip * Definitions.Fpy_norm(s, ps)) - Fpy_norm

# EQ 3 : Law of motion for K_p
loss_dict['eq_3']  =   lambdaf * lambdazy_norm * yzy_norm * sy_norm + E_t(lambda s, ps: beta * xip * (Definitions.piy_norm(s, ps)**iota_p * piy_norm**(1-iota_p) * pibar**(1-iota_p-(1-iota_p)) / Definitions.piy_norm(s, ps))**(lambdaf/(1-lambdaf)) * Definitions.Kpy_norm(s, ps) ) - Kpy_norm

# EQ 4 : Law of motion for K_p
loss_dict['eq_4']  =  Kpy_norm - (Fpy_norm * ((1 - xip * (pitilde / piy_norm)**(1/(1-lambdaf))) / (1 - xip))**(1-lambdaf))

# EQ 5: Law of motion for \latex{F_w}
loss_dict['eq_5'] = lambdazy_norm*PolicyState.wstary(policy_state)**(lambdaw/(lambdaw-1)) * hy_norm * (1 - taul) / lambdaw + E_t(lambda s, ps: beta * xiw * muzstar_p**((1-iotamu_p)/(1-lambdaw)) * (State.muzstarx(s)**(iotamu_p/(1-lambdaw)-1)) * (Definitions.piy_norm(s, ps)**iotaw_p * piy_norm**(1-iotaw_p) * pibar**(1-iotaw_p-(1-iotaw_p)))**(1/(1-lambdaw)) / Definitions.piy_norm(s, ps) * (1 / (Definitions.piy_norm(s, ps) * State.muzstarx(s) * Definitions.wtildey_norm(s, ps) / wtildey_norm))**(lambdaw/(1-lambdaw))  *  Definitions.Fwy_norm(s, ps) ) - Fwy_norm

# EQ 6: Law of motion for K_w
loss_dict['eq_6'] = (PolicyState.wstary(policy_state)**(lambdaw/(lambdaw-1)) * hy_norm)**(1+sigmaL) + E_t(lambda s, ps: beta * xiw * (Definitions.piy_norm(s, ps)**iotaw_p * piy_norm**(1-iotaw_p) * pibar**(1-iotaw_p-(1-iotaw_p)) / (Definitions.piy_norm(s, ps) * State.muzstarx(s) * Definitions.wtildey_norm(s, ps) / wtildey_norm) * muzstar_p**(1-iotamu_p) * State.muzstarx(s)**iotamu_p)**(lambdaw*(1+sigmaL)/(1-lambdaw)) * Definitions.Kwy_norm(s, ps)) - Kwy_norm

# EQ 7: Law of motion for w^*
loss_dict['eq_7']  = PolicyState.wstary(policy_state) - ((1 - xiw) * ( ((1 - xiw * (pitildew / piw * muzstar_p**(1-iotamu_p) * State.muzstarx(state)**iotamu_p)**(1/(1-lambdaw)))/ (1 - xiw))**lambdaw) + xiw * (pitildew / piw * muzstar_p**(1-iotamu_p) * State.muzstarx(state)**iotamu_p * State.wstarx(state))**(lambdaw/(1-lambdaw)))**(1/(lambdaw/(1-lambdaw)));

# EQ 8: Law of motion for Kwy
loss_dict['eq_8'] =  Kwy_norm - (((1 - xiw * (pitildew / piw * muzstar_p**(1-iotamu_p) * State.muzstarx(state)**iotamu_p)**(1/(1-lambdaw)))/ (1 - xiw))**(1-lambdaw*(1+sigmaL)) * wtildey_norm * Fwy_norm / psiL)

# EQ 9: Household FOC w.r.t. consumption
loss_dict['eq_9'] =  E_t(lambda s, ps:(1+tauc)*lambdazy_norm*Definitions.cy_norm(s, ps)*cy_norm-State.muzstarx(state)*Definitions.cy_norm(s, ps))+beta*b_p*cy_norm

# EQ 10: Household FOC w.r.t. risk-free bonds
loss_dict['eq_10'] = E_t(lambda s, ps: beta  *  Definitions.lambdazy_norm(s, ps) / (State.muzstarx(s) * Definitions.piy_norm(s, ps)) * (1 + Rey_norm) ) -  lambdazy_norm

# EQ 11: Household FOC w.r.t. investment
loss_dict['eq_11'] = - lambdazy_norm / State.muupx(state)  + lambdazy_norm * PolicyState.qy(policy_state) * (-spr_invest *  PolicyState.iy(policy_state) * State.muzstarx(state) * upsil + 1 - s_invest) + beta *  E_t(lambda s, ps: Definitions.lambdazy_norm(s, ps) * PolicyState.qy(ps)/ (State.muzstarx(s) * upsil))* ( PolicyState.iy(ps) * State.muzstarx(s) * upsil)**2  * (sqrt(Sdoupr_p / 2) * (tf.math.exp(sqrt(Sdoupr_p / 2) * ( PolicyState.muzstary(ps) * upsil * PolicyState.iy(ps) - muzstar_p * upsil))- tf.math.exp(-sqrt(Sdoupr_p / 2) * (PolicyState.muzstary(ps) * upsil * PolicyState.iy(ps) - muzstar_p * upsil))))

# EQ 12: Definition of return of entrepreneurs, Rk
loss_dict['eq_12'] = 1+Rky_norm - ((1 - tauk) * (PolicyState.uy(policy_state) * rentky_norm - autil) - (1 - delta) * PolicyState.qy(policy_state)) * piy_norm / (upsil * State.qx(state)) + tauk * delta

# EQ 13: Zero profit condition
loss_dict['eq_13'] = State.qx(state) * kbarx * (1 + Rky_norm) * ((1 - mu) * Gomegabar + omegabary_norm*tf.math.exp(State.dx(state)*State.thetax(state)) * (1 - Fomegabar)) / (State.nx(state) * (1 + State.Rex(state))) - State.qx(state) * kbarx / State.nx(state) + 1

# EQ 14: FOC for capital
loss_dict['eq_14'] = E_t(lambda s, ps: (1 + Definitions.Rky_norm(s, ps)) / (1 + Rey_norm) * ((1-p_d)*(1 - Definitions.omegabary_norm(s, ps)*(1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma)) -  norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma - sigma))+p_d*tf.math.exp(-State.thetax(s))*(1-Definitions.omegabary_norm(s, ps)*tf.math.exp(State.thetax(s)) *(1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+ State.thetax(s) + sigma**2 / 2) / sigma)) - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+ State.thetax(s) + sigma**2 / 2) / sigma - sigma)))+((1-p_d)*(1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma))+p_d*(1 -norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+ State.thetax(s) + sigma**2 / 2) / sigma)))/((1-p_d)*((1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma)) - mu/sigma * norm.pdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma))+p_d*((1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+tf.math.exp(State.thetax(s)) + sigma**2 / 2) / sigma)) - mu /sigma * norm.pdf((tf.math.log(Definitions.omegabary_norm(s, ps))+tf.math.exp(State.thetax(s)) + sigma**2 / 2) / sigma)))* ((1 + Definitions.Rky_norm(s, ps)) / (1 + Rey_norm)*((1-p_d)*(Definitions.omegabary_norm(s, ps) *(1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma))+ (1-mu)*norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps)) + sigma**2 / 2) / sigma - sigma) ) + p_d*tf.math.exp(-State.thetax(s))*(Definitions.omegabary_norm(s, ps)*tf.math.exp(State.thetax(s)) *( 1 - norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+ tf.math.exp(State.thetax(s))+ sigma**2 / 2) / sigma))+ (1-mu)*norm.cdf((tf.math.log(Definitions.omegabary_norm(s, ps))+ tf.math.exp(State.thetax(s)) + sigma**2 / 2) / sigma - sigma)))- 1))


# EQ 15: Efficiency condition for setting capital utilization
loss_dict['eq_15'] = rentky_norm - rentky_ss * np.exp(sigmaa*(PolicyState.uy(policy_state)-1))

# EQ 16 : Rental rate on capital
loss_dict['eq_16'] = rentky_norm - alpha*State.epsilx(state)*((upsil * State.muzstarx(state) * hy_norm * PolicyState.wstary(policy_state)**(lambdaw/(lambdaw-1)) /(PolicyState.uy(policy_state) * kbarx))**(1 - alpha)) * sy_norm

# EQ 17 : Marginal cost
loss_dict['eq_17'] = sy_norm-(rentky_norm / alpha)**alpha * (wtildey_norm / (1 - alpha))**(1-alpha) / State.epsilx(state)

# EQ 18 : production
loss_dict['eq_18'] = yzy_norm-(PolicyState.pstary(policy_state)**(lambdaf/(lambdaf-1)) * (State.epsilx(state) * (PolicyState.uy(policy_state) * State.kbarx(state) / (State.muzstarx(state) * upsil))**alpha * (hy_norm * PolicyState.wstary(policy_state)**(lambdaw/(lambdaw-1)))**(1-alpha) - phi))

# EQ 19 : Ressource constraint
loss_dict['eq_19'] = eta_g/(1-eta_g)*(cy_norm + PolicyState.iy(policy_state)) + cy_norm + PolicyState.iy(policy_state) / State.muupx(state) + autil * kbarx / (State.muzstarx(state) * upsil) + mu*Gomegabar*Rky_norm*State.qx(state)*kbarx/(piy_norm*State.muzstarx(state)) + bigtheta_p * (1 - gamma) * (ny_norm - we_p) / gamma - yzy_norm

# EQ 20: Monetary Policy Rule
loss_dict['eq_20'] = Rey_norm - ((1.0 - rhotil_p) * Re_p + rhotil_p * State.Rex(state) +  (1.0 - rhotil_p) * (aptil_p * (piy_norm - pibar) + adytil_p * (tf.math.log(yzy_norm) - tf.math.log(PolicyState.yzx(state)))))+State.epspx(state)


# EQ 21: Law of motion for capital
loss_dict['eq_21'] = kstary_norm - (1 - delta) * kbarx / (State.muzstarx(state) * upsil) - (1 - s_invest) * PolicyState.iy(policy_state)

# EQ  22: Law of motion of net worth
loss_dict['eq_22']  = gamma / (piy_norm * State.muzstarx(state)) * (Rky_norm - State.Rex(state)-((Gomegabar + omegabary_norm * (1 - Fomegabar)) - ((1 - mu) * Gomegabar + omegabary_norm * (1 - Fomegabar))) * (1 + Rky_norm)) * kbarx * State.qx(state) + we_p + gamma * (1 + State.Rex(state)) * State.nx(state) / (piy_norm * State.muzstarx(state)) - ny_norm ;


return loss_dict
