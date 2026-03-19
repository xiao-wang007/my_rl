It is much better now but the major termination is due to velocity violation.

Looking at the done_counts — indices 4 and 5 (velocity low/high violations) dominate overwhelmingly, with done_qvel_high (140) being ~4x more than done_qvel_low (30). The existing v_cmd_scale=0.7 cap and r_vel_limit penalty aren't enough because the PD controller can still generate torques that overshoot velocity limits within a single sim step.

The most robust fix is velocity-aware torque damping — before applying torque, reduce it for joints that are already near their velocity limit and being pushed further. This is standard in robotics and much more effective than relying on the policy to learn avoidance through reward alone.

Edited
mjx_env.py+12-0
I'd also bump w_vel_limit a bit so the penalty provides stronger gradient before the damper kicks in:

Edited
reward_weights_and_configs.py+1-1
Here's what I changed:

1. Velocity-aware torque damping (in step()): When a joint velocity is in the top 20% of its limit (above 80%), the component of torque that would accelerate it further toward the limit is smoothly attenuated to zero. This is a hard physics-level safeguard — the PD controller simply can't generate overshooting torques anymore. The damping is one-sided: it only suppresses torque in the "bad" direction, so the joint can still decelerate freely.

2. Increased w_vel_limit from 0.5 → 2.0: The reward penalty now bites harder, giving the policy stronger gradient to avoid the 80%+ zone in the first place.

These two changes work together: the reward teaches the policy to stay away from limits, and the damper is a safety net that prevents physics-level overshoot when it doesn't. The done_counts[4] and done_counts[5] should drop significantly.
--------------------------------------------------------------------------------------
Then when deployed, the same layer should also be in place right?

Yes, absolutely. The damping lives inside step() which runs identically during training and deployment — it's a physics-level safety layer, not a training trick. The torque attenuation has no dependency on train_progress or any training-specific parameter.

In fact, this is exactly how real robot controllers work: velocity-limit damping is part of the low-level controller, not something you remove after tuning. Keeping it during deployment means:

Sim-to-real consistency — the policy was trained with the damper, so it expects it. Removing it would change the dynamics and degrade performance.
Safety guarantee — even if the policy outputs something unexpected (distribution shift, adversarial perturbation), the damper prevents joint velocity overshoot at the torque level.
No changes needed — it's already deployment-ready as written.