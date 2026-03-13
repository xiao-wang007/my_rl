"""Quick check: subtree_linvel vs Jacobian vs cvel+cross for the EE site.
Uses plain MuJoCo (C backend) to avoid MJX compilation overhead.
The velocity math is identical between MuJoCo and MJX.
"""
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path(
    "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
)
m.opt.solver = mujoco.mjtSolver.mjSOL_CG
m.opt.timestep = 0.004

ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
ee_body_id = m.site_bodyid[ee_site_id]
m.site_pos[ee_site_id] = [0.0, 0.0, 0.15]
print(f"ee_site_id={ee_site_id}, ee_body_id={ee_body_id}")
print(f"site_pos (local) = {m.site_pos[ee_site_id]}")
print(f"nbody={m.nbody}, nv={m.nv}")

d = mujoco.MjData(m)

# non-trivial state so velocities aren't zero
d.qpos[:] = [0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3]
d.qvel[:] = [0.5, -0.8, 0.3, 1.0, -0.5, 1.5, -1.0]
mujoco.mj_forward(m, d)

# --- Method 1: Jacobian at site point (ground truth) ---
jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))
mujoco.mj_jac(m, d, jacp, jacr, d.site_xpos[ee_site_id], ee_body_id)
v_jac = jacp @ d.qvel  # (3, nv) @ (nv,) -> (3,)
print(f"\nv_site (Jacobian):        {v_jac}")

# --- Method 2: subtree_linvel (subtree CoM velocity) ---
v_sub = d.subtree_linvel[ee_body_id].copy()
print(f"v_subtree_linvel:         {v_sub}")

# --- Method 3: cvel (body frame 6D) + cross-product correction ---
# MuJoCo cvel layout: [wx, wy, wz, vx, vy, vz] in world frame
omega = d.cvel[ee_body_id, :3].copy()
v_body_origin = d.cvel[ee_body_id, 3:].copy()
R_body = d.xmat[ee_body_id].reshape(3, 3)
r_local = np.array([0.0, 0.0, 0.15])
r_world = R_body @ r_local
v_corrected = v_body_origin + np.cross(omega, r_world)
print(f"v_cvel+cross (corrected): {v_corrected}")

# --- Errors ---
err_sub = np.linalg.norm(v_jac - v_sub)
err_corrected = np.linalg.norm(v_jac - v_corrected)
omega_norm = np.linalg.norm(omega)
print(f"\n||v_jac - v_subtree||     = {err_sub:.6f} m/s")
print(f"||v_jac - v_corrected||   = {err_corrected:.6f} m/s")
print(f"||omega||                 = {omega_norm:.4f} rad/s")
print(f"max cross-term |w|*0.15  = {omega_norm*0.15:.4f} m/s")
print(f"v_mid target range       = [-0.5 .. 0.5] m/s")
print(f"cross-term / v_mid_max   = {omega_norm*0.15/0.5*100:.1f}%")
