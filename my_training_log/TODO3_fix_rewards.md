Here's the diagnosis. There are several issues with the reward design that are limiting learning:

Problem 1: Exponential rewards with tiny alphas are nearly flat
Your alpha3=0.01, alpha4=0.01 mean the reward kernels are extremely wide:

r_vel_mid = exp(-0.01 * vel_err²) — even when vel_err = 5 m/s, r_vel_mid = exp(-0.25) ≈ 0.78. The signal barely distinguishes good from bad.
r_retract = exp(-0.01 * retract_err²) — even 2m away gives exp(-0.04) ≈ 0.96. The policy gets nearly full retract reward no matter where it is.
Problem 2: Gating kills gradients
Everything is gated by r_pos_mid:

r_vel_gated = r_pos_mid * r_vel_mid
r_tilt_gated = r_pos_mid * r_tilt
r_dir = r_pos_mid * ...
With alpha1=1.0, if the EE is >0.5m from target, r_pos_mid < 0.78. All the velocity/tilt/direction signals are suppressed. The policy has to solve position first before it can learn anything else, but position reward alone (w_pos_mid=1.0) with alpha1=1.0 provides a very gentle gradient in task space.

Problem 3: Binary phase switch creates a cliff
reward = jnp.where(mid_done, r_retract, r_mid) — the moment mid_done flips, the reward function changes completely. The policy has no smooth signal guiding it from "approaching mid" to "retracting". And once mid_done is True, the mid-phase rewards vanish entirely.

Problem 4: mid target is very hard to achieve
eps_pos=0.05 (5cm) and eps_vel=0.1 m/s — the EE has to simultaneously be within 5cm of [0.4, 0.4, 0.05] AND match the random target velocity within 0.1 m/s. This is a very tight conjunction that might rarely trigger, meaning the policy never sees Phase 2 rewards.

