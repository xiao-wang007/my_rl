
What I meant is: a feedforward PPO policy chooses actions from the current observation only, basically a_t ~ pi(o_t). It has no memory. So if the true task state depends on hidden history, like mid_done persisting after the mid-target is achieved, then the policy is acting in a partially observed problem.

That is risky here because the reward switches phase based on mid_done in mjx_env.py (line 380) and mjx_env.py (line 386), but a feedforward policy only sees the current obs. If mid_done is not reconstructable from current observation alone, then the same-looking observation can require different behavior depending on hidden history.

There are alternatives:

Make the observation Markov. This is the simplest and usually best option here: include x_mid, v_mid, x_final, and mid_done in obs. Then plain feedforward PPO is fine.
Recurrent PPO. Use a GRU/LSTM policy so the network can remember past events. Your repo already has a GRU-based PPO in ppo_rnn.py (line 17), but that implementation is for discrete actions, so adapting it to Franka would require a continuous-action Gaussian head.
Frame stacking / short history. Concatenate the last few observations/actions. This is a lighter-weight compromise, but it’s weaker than explicitly exposing the hidden task variables.
Transformer / memory policies. Possible, but much more complexity than you need here.
For your env, I would not reach for recurrent PPO first. If the target and phase flag are part of the task definition, the cleanest solution is to expose them directly and keep feedforward PPO. RNNs are more useful when the hidden information is unavoidable, not when we can simply add it to the observation.

So the short answer is: yes, there are alternatives to feedforward PPO, but for your case the best move is probably not “switch algorithms”; it’s “fix the observation so the task is fully observable.”