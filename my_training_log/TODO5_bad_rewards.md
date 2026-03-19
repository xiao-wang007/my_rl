Here's what's limiting learning before any gain schedule issues:

Problem: The task is too hard for a from-scratch policy.

The reward only really activates near the target. With alpha1=10, r_pos_mid = exp(-10 * d²):

At 0.5m away: exp(-2.5) = 0.08 — almost zero reward
At 0.3m away: exp(-0.9) = 0.41 — still weak
At 0.1m away: exp(-0.1) = 0.90 — finally good
The policy starts from a fixed init_qpos and has to figure out, through random exploration, which joint velocity combinations move the EE toward [0.4, 0.4, 0.05]. With alpha1=10, if the initial EE position is >0.3m from target, the reward landscape is essentially flat — no gradient to follow.

The fix: add a distance-based shaping reward that gives credit for reducing the distance to the mid target, not just being close. This is the standard approach (potential-based reward shaping):

Edited
mjx_env.py+7-0
Now add it to r_mid and add the weight config:

What this does: r_approach = w_approach * (d_prev - d_now) / dt_env — positive reward when the EE moves closer to the mid target, negative when it moves away. Divided by dt_env to normalize the signal across timesteps.

This gives the policy an immediate, dense gradient from any starting distance — even 0.5m away, moving 1cm closer gives a tangible reward. Without it, the policy had to stumble randomly into the narrow exp(-10*d²) kernel to get any signal. The exponential proximity reward (r_pos_mid) still provides the final precision incentive once the policy is close.
----------------------------------------------------------------------------------------------------------------------------------
didn't my r_pos_mid already doing the same?

Compacted conversation

Good question — let me re-read the reward function to compare them precisely.

No, they do different things:

r_pos_mid = exp(-α₁ · ‖d‖²) is a state reward — it tells the agent "how close are you right now?" But with α₁=10, at 0.5m away: 
e
−
10
⋅
0.25
=
e
−
2.5
≈
0.08
e 
−10⋅0.25
 =e 
−2.5
 ≈0.08. At 0.3m: 
e
−
10
⋅
0.09
=
e
−
0.9
≈
0.41
e 
−10⋅0.09
 =e 
−0.9
 ≈0.41. The signal is nearly flat when far away — the gradient between 0.5m and 0.4m is tiny, so the agent gets almost no learning signal for moving in the right direction.

r_approach = (‖d_prev‖ - ‖d_now‖) / dt is a progress reward — it tells the agent "did you get closer this step?" This gives a clear, linear signal at any distance. Moving 1cm closer at 0.5m gives the same reward as moving 1cm closer at 0.1m.

The key difference: r_pos_mid has a vanishing gradient far from the target (exponential decay), while r_approach has a constant gradient everywhere. That's exactly why training plateaus — the agent never learns to approach because r_pos_mid looks the same whether it's 0.3m or 0.5m away.