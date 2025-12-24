from pydrake.all import AutoDiffXd
from pydrake.math import inv
import numpy as np



##############################################################################################
def Defects_q_ti_Euler(x_ti):
    global nq_arm, nv_arm, h
    q = x_ti[:nq_arm]
    q_next = x_ti[nq_arm:nq_arm+nq_arm]
    v_next = x_ti[nq_arm+nq_arm: nq_arm+nq_arm+nv_arm]

    y = q_next - q - h*v_next    

    return y

def Defects_v_ti_Euler(x_ti): 
    """
    0 = (Buₙ₊₁ + tau_g(qₙ₊₁) - C(qₙ₊₁, Vₙ₊₁) + ∑ᵢ (Jᵢ_WBᵀ(qₙ₊₁)ᵀ * Fᵢ_AB_W(λᵢ,ₙ₊₁))) * dt - M(qₙ₊₁) * (Vₙ₊₁ - Vₙ) 
    """
    global nv_arm, nq_arm, nu, h

    v = x_ti[:nv_arm] # 2d-cylinder + arm: 3 + 7 
    q_next = x_ti[nv_arm:nv_arm+nq_arm] 
    v_next = x_ti[nv_arm+nq_arm: nv_arm+nq_arm+nv_arm] 
    u_next = x_ti[nv_arm+nq_arm+nv_arm: nv_arm+nq_arm+nv_arm+nu]

    if x_ti.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        y = np.array([AutoDiffXd(0)]*nv_arm)

        if not AutoDiffArrayEqual(q_next, plant_eval.GetPositions(context_eval)):
            plant_eval.SetPositions(context_eval, q_next)
        if not AutoDiffArrayEqual(v_next, plant_eval.GetVelocities(context_eval)):
            plant_eval.SetVelocities(context_eval, v_next)

    else:
        plant_eval = plant_double
        context_eval = double_context
        y = np.zeros(nv_arm)

        if not np.array_equal(q_next, plant_eval.GetPositions(context_eval)):
            plant_eval.SetPositions(context_eval, q_next)
        if not np.array_equal(v_next, plant_eval.GetVelocities(context_eval)):
            plant_eval.SetVelocities(context_eval, v_next)
            
    # the actuation term
    B = plant_eval.MakeActuationMatrix()
    y += B @ u_next

    # C term
    c_next = plant_eval.CalcBiasTerm(context_eval)
    y -= c_next

    # G term
    g_next = plant_eval.CalcGravityGeneralizedForces(context_eval)
    y += g_next
    
    # acceleration term 
    M_next = plant_eval.CalcMassMatrix(context_eval)

    y = M_next @ (v_next - v) - y * h

    return y

##############################################################################################
def f_manipulator(x_ti):
    global nv_arm, nq_arm, nu
    q = x_ti[:nq_arm] 
    v = x_ti[nq_arm:nq_arm+nv_arm]
    u = x_ti[nq_arm+nv_arm: nq_arm+nv_arm+nu]
    if x_ti.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        y = np.array([AutoDiffXd(0)]*(nq_arm + nv_arm))
    else:
        plant_eval = plant_double
        context_eval = double_context
        y = np.zeros(nq_arm + nv_arm)

    plant_eval.SetPositions(context_eval, q)
    plant_eval.SetVelocities(context_eval, v)

    if x_ti.dtype == object:
        # for AD, we need to use the CalcMassMatrix method
        M_inv = inv(plant_eval.CalcMassMatrix(context_eval))
    else:
        # for double, we can use the CalcMassMatrix method directly
        M_inv = np.linalg.inv(plant_eval.CalcMassMatrix(context_eval))

    C = plant_eval.CalcBiasTerm(context_eval)
    G = plant_eval.CalcGravityGeneralizedForces(context_eval)
    y[:nq_arm] = v
    y[-nv_arm: ] = M_inv @ (u + G - C)

    return y

##############################################################################################
def f_manipulator_no_G(x_ti):
    global nv_arm, nq_arm, nu
    q = x_ti[:nq_arm] 
    v = x_ti[nq_arm:nq_arm+nv_arm]
    u = x_ti[nq_arm+nv_arm: nq_arm+nv_arm+nu]
    if x_ti.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        y = np.array([AutoDiffXd(0)]*(nq_arm + nv_arm))
    else:
        plant_eval = plant_double
        context_eval = double_context
        y = np.zeros(nq_arm + nv_arm)

    plant_eval.SetPositions(context_eval, q)
    plant_eval.SetVelocities(context_eval, v)

    if x_ti.dtype == object:
        # for AD, we need to use the CalcMassMatrix method
        M_inv = inv(plant_eval.CalcMassMatrix(context_eval))
    else:
        # for double, we can use the CalcMassMatrix method directly
        M_inv = np.linalg.inv(plant_eval.CalcMassMatrix(context_eval))

    C = plant_eval.CalcBiasTerm(context_eval)
    G = plant_eval.CalcGravityGeneralizedForces(context_eval)
    y[:nq_arm] = v
    y[-nv_arm: ] = M_inv @ (u - C)

    return y
##############################################################################################
def Defects_ti_HermiteSimpson(x_ti):
    global nv_arm, nq_arm, nu, h

    qv = x_ti[:nq_arm+nv_arm]
    u = x_ti[nq_arm+nv_arm: nq_arm+nv_arm+nu]
    qv_next = x_ti[nq_arm+nv_arm+nu: nq_arm+nv_arm+nu+nq_arm+nv_arm]
    u_next = x_ti[nq_arm+nv_arm+nu+nq_arm+nv_arm: nq_arm+nv_arm+nu+nq_arm+nv_arm+nu]

    if x_ti.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        plant_eval.SetPositionsAndVelocities(context_eval, qv_next)

    else:
        plant_eval = plant_double
        context_eval = double_context
        plant_eval.SetPositionsAndVelocities(context_eval, qv_next)
    
    f = f_manipulator(np.concatenate((qv, u)))
    f_next = f_manipulator(np.concatenate((qv_next, u_next)))
    x_mid = 0.5 * (qv + qv_next) + 0.125 * h * (f - f_next)
    u_mid = 0.5 * (u + u_next)
    f_mid = f_manipulator(np.concatenate((x_mid, u_mid)))

    return qv_next - qv - 1./6. * h * (f + 4 * f_mid + f_next)

##############################################################################################
def Defects_ti_HermiteSimpson_no_G(x_ti):
    global nv_arm, nq_arm, nu, h

    qv = x_ti[:nq_arm+nv_arm]
    u = x_ti[nq_arm+nv_arm: nq_arm+nv_arm+nu]
    qv_next = x_ti[nq_arm+nv_arm+nu: nq_arm+nv_arm+nu+nq_arm+nv_arm]
    u_next = x_ti[nq_arm+nv_arm+nu+nq_arm+nv_arm: nq_arm+nv_arm+nu+nq_arm+nv_arm+nu]

    if x_ti.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        plant_eval.SetPositionsAndVelocities(context_eval, qv_next)

    else:
        plant_eval = plant_double
        context_eval = double_context
        # if not np.array_equal(qv_next, plant_eval.GetPositionsAndVelocities(context_eval)):
        #     plant_eval.SetPositionsAndVelocities(context_eval, qv_next)
        plant_eval.SetPositionsAndVelocities(context_eval, qv_next)

    f = f_manipulator_no_G(np.concatenate((qv, u)))
    f_next = f_manipulator_no_G(np.concatenate((qv_next, u_next)))
    x_mid = 0.5 * (qv + qv_next) + 0.125 * h * (f - f_next)
    u_mid = 0.5 * (u + u_next)
    f_mid = f_manipulator_no_G(np.concatenate((x_mid, u_mid)))

    return qv_next - qv - 1./6. * h * (f + 4 * f_mid + f_next)

##############################################################################################