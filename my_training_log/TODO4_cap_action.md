my average length rollout/ep_len_mean is only about 13. Capping the action from the policy before turning it into torques could help

Read mjx_env.py, lines 275 to 310

Right — the policy outputs in [-1, 1] from tanh, which maps to the full velocity range. At the extremes, it's commanding max velocity, and the PD controller generates huge torques to track it. Capping the commanded velocity to a fraction of the limits keeps things safe:

Edited
mjx_env.py+8-1
This caps the commanded velocity to ±70% of the joint limits. The policy can still use its full [-1, 1] output range, but it maps to a safer velocity envelope. With joints 1-4 at ±2.175 rad/s, the max command is now ±1.52 rad/s — enough room for the PD controller to track without overshooting past the hard limits.

You might also want to cap action_v_prev the same way since it's used in the position reference q_ref:

Edited
mjx_env.py+1-1
Both velocity commands now use the same 70% cap. This should dramatically reduce velocity terminations and let episodes survive long enough for the policy to learn the actual task.