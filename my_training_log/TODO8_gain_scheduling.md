The performance drop is expected — you're suddenly giving the policy 14 new action dimensions (dkp, dkd) that it hasn't learned to use, and random outputs immediately destabilize the PD controller. The standard approaches:

Your current approach (linear ramp) is fine in principle, but the issue is:

dkp_max = 100 means kp can double (100→200). That's a massive perturbation.
The ramp from 0.6→1.0 means only 40% of training remains to learn 14 new dimensions.
Recommended changes:

Reduce dkp_max and dkd_max — start with small residual authority (e.g. ±20% of base gains), not ±100%. The policy shouldn't need to double its stiffness.

Start the gain schedule earlier — give more training time for adaptation. 0.3→0.8 is more reasonable.

Shrink the residual range first, widen later — this is the key insight. Rather than a binary on/off ramp, limit the magnitude the residual can reach early on.

Edited
mjx_env.py+8-6
Summary of changes and reasoning:

dkp_max: 100 → 20, dkd_max: 20 → 4 — The old range let kp go from 100 to 200 (2x!). Now it's 100→120 (~20% variation). This is plenty for impedance adaptation —
----------------------------------------------------------
I implemented a cleaner velocity-aware torque guard in mjx_env.py (line 174) and applied it in the torque path at mjx_env.py (line 324). It ramps on once joint speed exceeds 75% of the limit (vel_guard_start_ratio at mjx_env.py (line 137)), and it only suppresses the part of tau that would push the joint further into the limit. Braking torque is preserved.

That is the right pattern for RL robotics. The policy can still command aggressively, but the low-level controller refuses to accelerate into unsafe velocity regions. If you deploy this policy with the same controller stack, this guard should stay in deployment too, because it defines the actuator semantics the policy was trained against.

python -m py_compile mjx_rl/my_builds/mjx_env.py passed.

If qvel terminations are still dominant, the first knobs to tune are:

Lower vel_guard_start_ratio from 0.75 to 0.65 if you want the guard to engage earlier.
Lower v_cmd_scale from 0.7 to 0.6 if the policy is still asking for more speed than the plant can track cleanly.
If needed after that, we can switch from your current q_ref built from action_v_prev to action_in_v, which usually reduces one-step lag and overshoot.