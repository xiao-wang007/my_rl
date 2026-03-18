import numpy as np

r_configs1 = {
    "alpha1": np.float32(10.0),   # was 1.0; sharper position kernel so policy sees gradient
    "alpha2": np.float32(0.01),
    "alpha3": np.float32(1.0),    # was 0.01; velocity reward was nearly flat
    "alpha4": np.float32(1.0),    # was 0.01; retract reward was nearly flat
    "k": np.float32(0.1),
    "eps1": np.float32(0.05),     # cm
    "k_pos": np.float32(1.0),
    "k_vel": np.float32(1.0),
    "eps_pos_end": np.float32(0.05),  # final tight threshold (5cm)
    "eps_vel_end": np.float32(0.1),   # final tight threshold (m/s)
    "eps_pos_start": np.float32(0.20),  # easy threshold at start of training
    "eps_vel_start": np.float32(0.5),   # easy threshold at start of training
    "eps_tilt": np.float32(0.1),
    "alpha_tilt": np.float32(1.0),
    "beta_tilt": np.float32(2.0),
    "eps_vmag": np.float32(0.05), # m/s
    "eps_vel": np.float32(0.05),  # normalizing epsilon for velocity unit vectors
}

r_weights1 = {
   "w_pos_mid": np.float32(1.0),
   "w_vel_mid": np.float32(1.0),
   "w_pos_final": np.float32(1.0),
   "w_vel_progress": np.float32(0.05),
   "w_action_rate": np.float32(0.001),
   "w_tilt": np.float32(1.0),
   "w_vel_limit": np.float32(0.5),   # penalty for approaching joint velocity limits
}
