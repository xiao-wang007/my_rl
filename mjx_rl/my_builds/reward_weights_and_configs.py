import numpy as np

r_configs1 = {
    "alpha1": np.float32(1.0),
    "alpha2": np.float32(0.01),
    "alpha3": np.float32(0.01),
    "alpha4": np.float32(0.01),
    "k": np.float32(0.1),
    "eps1": np.float32(0.05), # cm
    "k_pos": np.float32(1.0),
    "k_vel": np.float32(1.0),
    "eps_pos": np.float32(0.05), # cm
    "eps_vel": np.float32(0.1), # m/s
    "eps_tilt": np.float32(0.1),
    "alpha_tilt": np.float32(1.0),
    "beta_tilt": np.float32(2.0),
    "eps_vmag": np.float32(0.05), # m/s
}

r_weights1 = {
   "w_pos_mid": np.float32(1.0),
   "w_vel_mid": np.float32(1.0),
   "w_pos_final": np.float32(1.0),
   "w_vel_progress": np.float32(0.05),
   "w_action_rate": np.float32(0.001),
   "w_tilt": np.float32(1.0)
}
