from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, DiagramBuilder_,  SceneGraph, SceneGraph_,
    FindResourceOrThrow, InverseDynamicsController, 
    MultibodyPlant, MultibodyPlant_, Parser, Simulator, InverseKinematics)

from pydrake.autodiffutils import ExtractGradient, InitializeAutoDiff
import pydrake.autodiffutils
from pydrake.symbolic import MakeVectorContinuousVariable

import pydrake
from pydrake import geometry
from pydrake.math import RigidTransform, RigidTransform_, RollPitchYaw, RollPitchYaw_, RotationMatrix, RotationMatrix_ 
from pydrake.math import inv
from pydrake.solvers import MathematicalProgram, Solve, IpoptSolver, SnoptSolver
from functools import partial
from pydrake.all import (
    JointIndex, PiecewisePolynomial, JacobianWrtVariable,
    eq, ge, le,  AutoDiffXd, SnoptSolver, IpoptSolver,  
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
    )
from pydrake.geometry import (MeshcatVisualizer, StartMeshcat, FrameId)
from pydrake.visualization import (VisualizationConfig, 
                                   ApplyVisualizationConfig, 
                                   AddDefaultVisualization, 
                                   AddFrameTriadIllustration,)
from pydrake.multibody.tree import PrismaticJoint, RevoluteJoint, BallRpyJoint

import numpy as np
from functools import partial
from collections import OrderedDict, namedtuple
from pydrake.multibody.tree import FixedOffsetFrame, SpatialInertia, UnitInertia

import sys

def AutoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))


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


def Defects_ti_HermiteSimpson(x_ti):
    global h, nv_arm, nq_arm, nu, plant_double, plant_AD, double_context, AD_context 

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


def eeTilted_cst(q, ahat, bhat):
    global plant_double, plant_AD, double_context, AD_context
    if q.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        plant_eval.SetPositions(context_eval, q)

    else:
        plant_eval = plant_double
        context_eval = double_context
        plant_eval.SetPositions(context_eval, q)

    ee_frame = plant_eval.GetFrameByName("dummy1")
    X_W_ee = ee_frame.CalcPoseInWorld(context_eval)
    zhat_ee_w = X_W_ee.rotation().matrix()[:, 2]

    return  np.array([np.dot(zhat_ee_w, ahat), np.dot(zhat_ee_w, bhat)])


def fix_p_ee_w(x, pt_ee_w):
    global plant_double, plant_AD, double_context, AD_context
    if x.dtype == object:
        plant_eval = plant_AD
        context_eval = AD_context
        if not AutoDiffArrayEqual(x, plant_eval.GetPositions(context_eval)):
            plant_eval.SetPositions(context_eval, x)

    else:
        plant_eval = plant_double
        context_eval = double_context
        if not np.array_equal(x, plant_eval.GetPositions(context_eval)):
            plant_eval.SetPositions(context_eval, x)
    
    ee_frame = plant_eval.GetFrameByName("dummy1")
    ee_pose_w = ee_frame.CalcPoseInWorld(context_eval)
    p_ee_w = ee_pose_w.translation()

    diff = p_ee_w - pt_ee_w
    y = diff[0]**2 + diff[1]**2 + diff[2]**2
    return np.array([y])