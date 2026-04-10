from scopt import *

franka_path = Path(__file__).parent.parent.parent / "mujoco_menagerie/franka_emika_panda/mjx_panda_nohand.xml"
print(f"franka_path = {franka_path} \n")

franka_model = load_franka_scopt_model(xml_path=franka_path)

#* not printing this here, there are too many options
# print(f"franka_model = {dir(franka_model.mj_model)}")

mj_model = franka_model.mj_model
q_low = mj_model.jnt_range[:, 0]
q_high = mj_model.jnt_range[:, 1]
v_robot_up = jnp.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
v_robot_low = -v_robot_up
tau_low = mj_model.actuator_ctrlrange[:, 0]
tau_high = mj_model.actuator_ctrlrange[:, 1]
coeff = 0.1

key = jax.random.PRNGKey(0)
key_q, key_v, key_tau = jax.random.split(key, 3)
q = jax.random.uniform(key_q, shape=q_low.shape, minval=q_low, maxval=q_high)
v = jax.random.uniform(key_v, shape=v_robot_low.shape, minval=v_robot_low, maxval=v_robot_up)
tau = jax.random.uniform(key_tau, shape=tau_low.shape, minval=coeff*tau_low, maxval=coeff*tau_high)
print(f"q = {q} \n")
print(f"v = {v} \n")
print(f"tau = {tau} \n")

#* this is continuous time 
# qddot = franka_qddot(model=franka_model, q=q, qdot=v, tau=tau)
# print(f"qddot = {qddot} \n")

# jacq, jacv, jacu = franka_dynamics_jacobians(model=franka_model, q=q, qdot=v, tau=tau)
# print(f"jacq shape = {jacq.shape} \n")
# print(f"jacv shape= {jacv.shape} \n")
# print(f"jacu shape = {jacu.shape} \n")

# A, B = franka_state_space_jacobians(model=franka_model, 
#                                     q=q, qdot=v, tau=tau, dt=0.02, method="rk4_linearize")

# print(f"A shape = {A.shape} \n"
#       f"B shape = {B.shape} \n")

# # jit the functions
# f_func, jac_func = make_jitted_dynamics(franka_model)
# _, AB_dis_fn = make_jitted_state_space_jacobians(franka_model, method="rk4_linearize")
# A_out, B_out = AB_dis_fn(q, v, tau, jnp.array(0.02, dtype=jnp.float32))
# print(f"A_out shape = {A_out.shape} \n"
#       f"B_out shape = {B_out.shape} \n")

mjx_model = franka_model.mjx_model
mjx_data = mjx.make_data(mjx_model)
# print(f"mjx_data = {mjx_data} \n")

ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")
p_ee_w_val = p_ee_w(q=q, model=mjx_model, data=mjx_data, ee_body_id=ee_body_id)
print(f"p_ee_w = {p_ee_w_val} \n")
v_ee_w_val = v_ee_w(q=q, qdot=v, model=mjx_model, data=mjx_data, ee_body_id=ee_body_id)
print(f"v_ee_w = {v_ee_w_val} \n")

p_ee_jac_w_val = p_ee_jac_w(q=q, model=mjx_model, data=mjx_data, ee_body_id=ee_body_id)
print(f"p_ee_jac_w = {p_ee_jac_w_val} \n")
print(f"p_ee_jac_w shape = {p_ee_jac_w_val.shape} \n")
v_ee_w1 = p_ee_jac_w_val.T @ v
print(f"v_ee_w1 = {v_ee_w1} \n")
print(f"||v_ee_w - v_ee_w1|| = {jnp.linalg.norm(v_ee_w_val - v_ee_w1)} \n")


# test mdir
d = jnp.array([0.1, 0.1, 0.0], dtype=jnp.float32)
inv_mdir = inv_m_dir(q=q, d=d, model=mjx_model, data=mjx_data, ee_body_id=ee_body_id)
print(f"inv_mdir = {inv_mdir} \n")

inv_mdir_grad = inv_m_dir_grad(q=q, d=d, model=mjx_model, data=mjx_data, ee_body_id=ee_body_id)
print(f"inv_mdir_grad = {inv_mdir_grad} \n")

