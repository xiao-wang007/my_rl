from src.visual_utils import *
import numpy as np
import time
import yaml
from pathlib import Path
from src import constraints as csts
from src.build_and_solve import (
    prog_init,
    prog_add_csts,
    prog_add_cost,
    prog_set_initial_guess,
    prog_solve,
)

q_init = np.array([0.6465, -0.4952, -0.0675, -3.,  0.6238,  2.572 ,  0.2666])
meshcat = StartMeshcat()
plant_vis, context_vis, sceneGraph_vis, diagram_vis = scene_visualizer(arm_pose=q_init, meshcat=meshcat, h=0.001)


if meshcat is not None:
    try:
        input("Open the Meshcat localhost page, then press Enter to start optimization...")
    except EOFError:
        pass


with (Path(__file__).resolve().parent / "params.yaml").open("r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

#* set global variables for constraints
csts.plant_double = plant_vis
csts.double_context = context_vis
csts.plant_AD = plant_vis.ToScalarType[AutoDiffXd]()
csts.AD_context = csts.plant_AD.CreateDefaultContext()
csts.nq_arm = plant_vis.num_positions()
csts.nv_arm = plant_vis.num_velocities()
csts.nu = plant_vis.num_actuators()

q_init_np = np.array(params["q_init"])
N = params["N"]
h = params["h"]
csts.h = h
maxiter = params["maxiter"]

prog, q_vars, v_vars, u_vars = prog_init(N=N, q0=q_init_np)
prog_add_csts(
    prog,
    q_vars,
    v_vars,
    u_vars,
    np.array(params["ahat"]),
    np.array(params["bhat"]),
    np.deg2rad(params["angle_at_tf"]),
    params["tilting_relax"],
    np.array(params["p_ee_target_w"]),
    params["p_ee_relaxation"],
)

prog_add_cost(prog, q_vars, u_vars)

prog_set_initial_guess(
    prog,
    q_vars,
    v_vars,
    u_vars,
    q_init_np
)

q_sol, v_sol, u_sol = prog_solve(prog, q_vars, v_vars, u_vars, maxiter=maxiter)


#* visualize qf
qf = q_sol[-1, :]
print("Final arm configuration at tf: \n", qf)

# hold to view
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
