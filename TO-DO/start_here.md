Love it. Let's build this from zero in a way that (1) will actually run, (2) won’t wreck the arm, and (3) will scale later to contact tasks.

I'll give you a concrete plan with 6 parts:

1. Define the task precisely
2. Choose the action space
3. Choose the observation/state
4. Design reward
5. Build the training stack (sim → SAC)
6. Safety + “gotchas” for Franka

I’ll assume you’re controlling a Franka Emika Panda, you want the end effector to move toward some Cartesian goal along a desired velocity profile, and you're using Drake / Gazebo sim first before the real arm (which matches your previous work).

---

1. Task definition

---

Goal:

* You give a target pose or target position in Cartesian space, call it ( x^* ) (position) or ((R^*, p^*)) (orientation + position).
* You also give a desired end-effector spatial velocity ( v^* ) (linear and maybe angular).
* The robot should move the end effector so that:

  * it's near the target pose,
  * and its current velocity matches the commanded velocity profile.

This is *not* pure “go to pose and stop.” It’s “track a task-space motion reference.” That’s good because it’s closer to dynamic manipulation.

We will train a policy (\pi_\theta(o_t) \to a_t) with SAC to minimize pose error and velocity error.

---

2. Action space (what SAC outputs)

---

You have three reasonable choices. This is important because it changes stability.

Option A. Joint velocity command (\dot{q}_{cmd})

* Policy outputs a 7D vector (for Panda’s 7 joints), clipped to safe limits.
* Low-level controller in sim (and later on the arm) takes that as the commanded velocity.

Pros: easy, smooth.
Cons: policy has to learn its own IK implicitly.

Option B. Task-space twist command (v_{ee,cmd})

* Policy outputs desired end-effector spatial velocity (e.g. ($v_x, v_y, v_z, \omega_x, \omega_y, \omega_z$)).
* You run an operational-space/IK controller that turns that twist into joint velocities/torques.

Pros: very aligned with “reach a Cartesian target at a velocity.”
Cons: you must provide a stable differential IK layer every step.

Option C. Torque (\tau)

* Policy outputs 7 joint torques.

Pros: maximal expressivity.
Cons: easiest way to blow up your robot, hardest to learn, super unstable around contacts unless you wrap it in impedance.

Given you’re starting from scratch on a reaching / velocity tracking task (free-space, no hard contact), the most practical first step is:

→ Start with **Option A: joint velocity commands**.
You can absolutely graduate later to torques + impedance shaping, but velocity control will get you a working SAC loop much faster. It also plays nicely with Gazebo/Drake and Franka Control Interface (FCI)-style velocity limits.

---

3. Observation (policy input)

---

Your observation vector at time (t) should let the policy infer:

* where am I,
* where should I be moving,
* how far off my motion is.

A solid minimal observation:

1. Robot state

   * ( q_t \in \mathbb{R}^7 )  (joint angles)
   * ( \dot{q}_t \in \mathbb{R}^7 ) (joint velocities)

2. End-effector task state

   * ( p_t \in \mathbb{R}^3 ): current end-effector position in base frame
   * ( R_t ): current orientation. Represent as 3D axis-angle or 6D continuous rep; don’t give raw quaternions directly without care about sign flip.
   * ( v_{ee,t}^{lin} \in \mathbb{R}^3 ): current linear velocity of the EE in base frame
   * ( v_{ee,t}^{ang} \in \mathbb{R}^3 ): current angular velocity

3. Goal / command

   * ( p^* \in \mathbb{R}^3 ): target Cartesian position
   * orientation target, e.g. ( R^* ) as axis-angle
   * desired EE linear vel ( v^{*}_{lin} \in \mathbb{R}^3 )
   * desired EE angular vel ( v^{*}_{ang} \in \mathbb{R}^3 )

4. Errors (this really helps SAC learn faster)

   * ( e_p = p^* - p_t )  (position error)
   * ( e_R ): orientation error in some minimal 3D form (log map of (R_t^\top R^*))
   * ( e_v^{lin} = v^{*}*{lin} - v^{lin}*{ee,t} )
   * ( e_v^{ang} = v^{*}*{ang} - v^{ang}*{ee,t} )

You can either feed raw states + targets and let the net subtract, or explicitly include the precomputed error terms. Including the error terms is basically giving it “proportional control structure for free,” which massively reduces sample complexity.

So your observation could be a single concatenated float vector:
[ o_t = [q, \dot{q}, e_p, e_R, e_v^{lin}, e_v^{ang}, p_t, v_{ee,t}^{lin}, v_{ee,t}^{ang}] ]

That’s ~7 + 7 + (3+3+3+3) + 3 + 3 + 3 ≈ 35-40 dims. That’s fine for SAC.

---

4. Reward design

---

This is the heart. We give SAC dense shaping so it doesn’t have to “get lucky.”

At each timestep (t), define:

* Position tracking cost
  ( c_p = \lVert e_p \rVert^2 )

* Orientation tracking cost
  ( c_R = \lVert e_R \rVert^2 )

* Velocity tracking cost
  ( c_v = \lVert e_v^{lin} \rVert^2 + \beta \lVert e_v^{ang} \rVert^2 )

* Action smoothness / effort cost
  ( c_a = \lVert \dot{q}_{cmd} \rVert^2 )  (penalize crazy joint velocities)

* Joint limit margin cost (safety shaping)
  For each joint i, penalize being too close to hard limits:
  ( c_{lim} = \sum_i \text{softplus}( |q_i - q_i^{mid}| - m ) )
  where (q_i^{mid}) is the midrange between min/max and m is allowed margin.
  This gently discourages stretching to hardware limits.

Then make reward = negative weighted sum:
[
r_t =

* w_p c_p
* w_R c_R
* w_v c_v
* w_a c_a
* w_{lim} c_{lim}

- r_{\text{bonus}}
  ]

with an optional sparse-ish success bonus:

* ( r_{\text{bonus}} = r_{succ} ) if (|e_p| < \epsilon_p) **and** (|e_v^{lin}| < \epsilon_v).
  (Gives the policy a “yes good keep doing that” anchor.)

Good starting weights (tune later):

* (w_p) high (e.g. 5–10)
* (w_v) medium (e.g. 1–3)
* (w_R) small at first if you don’t care about orientation yet
* (w_a) small (0.01–0.1)
* (w_{lim}) small (0.1)

Intuition:

* Early in training, the agent mostly learns “move the EE roughly toward the goal.”
* As SAC improves, it naturally starts caring about velocity tracking because that term matters every timestep.

---

5. Training stack

---

Here’s the loop you actually build:

Step 0. Sim setup

* Load Franka Panda URDF/SDF into Drake or Gazebo.
* Gravity on.
* No obstacles yet.
* Add a controller shim that:

  * receives joint velocity commands from the RL policy every control step,
  * clips them to safe limits,
  * integrates them through the plant (or sends to Gazebo joint velocity interface).

Step 1. Environment (Gym-style)
Your env needs:

* `reset()`:

  * Randomize initial joint configuration within a safe, collision-free posture.
  * Sample a random target pose (p^*, R^*).
  * Sample a desired Cartesian velocity profile (v^*) (e.g. “move +x at 5 cm/s”).
  * Return initial observation vector (o_0).

* `step(action)`:

  * Interpret `action` as joint velocity command.
  * Clip to per-joint max vel (Franka has ~2 rad/s joints but you’ll want <0.4 rad/s early).
  * Simulate ∆t (e.g. 5 ms, 10 ms).
  * Compute new obs, reward, done flag.
  * “done” when episode length hits horizon (e.g. 2 seconds of motion) or if anything unsafe.

Important: keep the episode horizon short at first (like 200 steps @10 ms = 2 s). Short horizons stabilize credit assignment a lot.

Step 2. SAC agent

* Use standard SAC with:

  * Actor: 2-layer MLP, 256 units ReLU each layer.
  * Critic: twin Q networks (as in the SAC paper you gave me, which trains two Qθ networks and uses the min to reduce overestimation bias).
  * Replay buffer: big ring buffer (e.g. 1e6 transitions).
  * Discount γ = 0.99.
  * Learning rate ~3e-4 Adam for actor and critics. (Matches the SAC hyperparams in Table 1 of the paper you shared. )
  * Target network smoothing τ = 0.005.

* Action squashing:

  * Actor outputs unconstrained Gaussian.
  * Apply tanh() to squash to [-1,1].
  * Scale that to your per-joint max vel.
    This matches SAC’s trick of using tanh and correcting the log-prob with the change-of-variables term so gradients are right.

Step 3. Curriculum (optional, but helps)
Early training can be brutal if you randomize huge targets. Do this:

* Stage 1: goal is always near the current EE position (like within 5 cm), and desired velocity is slow.
* Stage 2: larger workspace, faster desired velocities.
* Stage 3: full reachable workspace of the arm.

All you’re doing is widening the randomization ranges after the policy reaches a performance threshold. SAC benefits a lot from this kind of curriculum because it reuses replay from easier stages to bootstrap harder stages efficiently. (Replay buffer off-policy training is a natural fit for curriculum.)

Step 4. Domain randomization (prep for real robot)
Before transferring to hardware, start randomizing:

* link masses / inertias ±5–10%
* joint damping/friction
* a little sensor noise on q, (\dot{q}), and especially EE pose
  This helps sim-to-real.

Step 5. Transfer to the real Panda
When you go on real hardware:

* Keep the same policy architecture.
* Wrap the learned policy with:

  * velocity clamp (absolute max per joint),
  * workspace clamp (if EE leaves a safe box relative to base, you override with zero velocity),
  * rate limiter on velocity changes (slew limit).
* Run at a safe low gain first, i.e. scale the commanded joint velocities by maybe 0.25 or 0.1 to start. You can always scale up later.

This gives you a “safety shell” around SAC so the first real execution doesn’t yeet the arm.

---

6. Subtle but important gotchas

---

1. Don’t give it torque control on day one.
   Velocity control in joint space is dramatically more forgiving.

2. Always include velocity tracking error in the reward if you care about following a commanded Cartesian velocity.
   If you only reward position error, it will just go to the goal and park, not move with the requested profile.

3. Orientation:
   If orientation isn’t important initially, don’t reward it at first. You can add that term later in the curriculum. Otherwise the policy wastes capacity solving a harder 6-DoF task than you actually need.

4. Episode termination signal is a teacher.
   If you terminate early on “bad states,” SAC will learn very fast to avoid them because early termination cuts off future reward. You can exploit this as a soft safety shield in sim.

5. Tune reward weights before you obsess over SAC hyperparams.
   SAC itself is usually stable with the defaults from the paper (twin Qs, entropy temperature α either fixed or learned). Most “it doesn’t learn” issues are actually reward shaping / observation mistakes, not SAC bugs.

6. Log everything in world frame.
   You’ll want to plot:

   * |e_p| over time
   * |e_v| over time
   * action magnitudes
     per episode.
     That’s how you’ll know if you’re making progress.

---

## Quick recipe / checklist you can literally execute next

* [ ] Build a Gym-style env around your Franka sim that:
  - steps joint velocities
  - exposes EE pose, EE velocity, goals, and errors in the obs
  - computes the dense reward above

* [ ] Plug that env into a vanilla SAC implementation (e.g. your own PyTorch or an existing one). Make sure it supports tanh-squashed Gaussian actions with log-prob correction like SAC. (That's standard SAC.)

* [ ] Start training with tiny workspaces (goal 5 cm away, slow desired velocity), short horizons, no orientation tracking.

* [ ] Once it can do smooth point-to-point motion at a commanded velocity in sim, expand the workspace and speed, start randomizing physics, and prepare for real deployment with clamps.

That’s your from-scratch path.

If you want, I can draft the exact observation dict, reward function, and action scaling in PyTorch-style pseudocode next.
