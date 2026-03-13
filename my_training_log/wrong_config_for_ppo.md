Yes, for PPO specifically it can hurt learning in two ways:

1. Stale policy problem. PPO's clipped surrogate objective assumes the behavior policy (that collected the data) is close to the current policy. With NUM_STEPS = 200, you collect 200 transitions under one policy, then update. But with UPDATE_EPOCHS = 1 and NUM_MINIBATCHES = 32, you only do 32 gradient steps on that data — the policy barely changes, so you under-utilize the data. If you increased UPDATE_EPOCHS to compensate, the policy drifts far from the collection policy by the later epochs, making the importance-sampling ratio unreliable (even with clipping).

2. GAE bias-variance tradeoff shifts. With NUM_STEPS = 200 at 50Hz control, each rollout spans 4 seconds of sim time — that's a large chunk of your 1000-step (20s) episode. The advantage estimates for early steps in the rollout depend on a long chain of future rewards, increasing variance. Shorter rollouts (10–20 steps) give tighter, lower-variance advantage estimates bootstrapped from the value function, which is what PPO is designed for.

3. Gradient staleness within an update. The 200 transitions are collected sequentially with the same network weights. A shorter rollout means you update the policy more frequently relative to experience collected, so the policy improves faster per wall-clock unit.

The sweet spot for MJX/Brax-style PPO is typically NUM_STEPS = 5–20. Your current 200 is ~10–40× too long. Combined with UPDATE_EPOCHS = 1, you're collecting a lot of data but barely learning from it — worst of both worlds.

The config change I recommended (NUM_STEPS = 20, UPDATE_EPOCHS = 4) fixes both speed and learning efficiency: shorter rollouts with more reuse.