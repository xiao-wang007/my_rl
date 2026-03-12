import jax.numpy as jnp

r_configs_1 = {
    "alpha1": jnp.float32(1.0),
    "alpha2": jnp.float32(0.01),
    "alpha3": jnp.float32(0.01),
    "alpha4": jnp.float32(0.01),
    "k": jnp.float32(0.1),
    "eps1": jnp.float32(0.05), # cm
    "k_pos": jnp.float32(1.0),
    "k_vel": jnp.float32(1.0),
    "eps_pos": jnp.float32(0.05), # cm
    "eps_vel": jnp.float32(0.1), # m/s
    "eps_tilt": jnp.float32(0.1),
    "alpha_tilt": jnp.float32(1.0),
    "beta_tilt": jnp.float32(2.0),
    "eps_vmag": jnp.float32(0.05), # m/s
}

r_weights1 = {
   "w_pos_mid": jnp.float32(1.0),
   "w_vel_mid": jnp.float32(1.0),
   "w_pos_final": jnp.float32(1.0),
   "w_vel_progress": jnp.float32(0.05),
   "w_action_rate": jnp.float32(0.001),
   "w_tilt": jnp.float32(1.0)
}
