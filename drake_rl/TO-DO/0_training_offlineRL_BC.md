It depends a lot on **state/action dimension, horizon length, and how “wide” the behavior you want is**, so there isn’t one universal number. But you *can* plan this with practical rules of thumb that work well in robotics.

Below are **useful ranges** for **BC + offline-safe RL (IQL/AWAC/CQL)** when the dataset is coming from **trajectory optimization / MPC**.

---

## What you should count: transitions, not “trajectories”

In RL training, “data” is usually counted as **transitions**:
[
(s_t, a_t, r_t, s_{t+1})
]
Let:

* (N_{\text{traj}}) = number of trajectories
* (H) = average horizon length (timesteps)
* Then transitions (\approx N_{\text{traj}} \times H)

---

## Practical ranges that usually work

### A) Low-dim control (classic MuJoCo-style)

Examples: 4–20D states, 1–8D actions, smooth dynamics
**BC**: ~(10^4)–(10^5) transitions
**Offline RL (IQL/AWAC/CQL)**: ~(10^5)–(10^6) transitions

This is why many papers report “100k–1M steps”.

---

### B) Robot manipulation with contact (7-DoF arm + gripper)

Even with TO-generated data, contact makes generalization harder.

**BC**: ~(10^5)–(10^6) transitions
**Offline-safe RL**: ~(5\times 10^5)–(5\times 10^6) transitions

If you only have (10^4)–(10^5) transitions, you can get “it moves”, but robustness will be poor.

---

### C) If your observation is images (or high-dimensional features)

If you train perception + control end-to-end, data needs jump by ~10×–100×.

**BC**: ~(10^6)–(10^7) transitions
**Offline-safe RL**: ~(10^7+) transitions

If you use a frozen encoder / pretrained vision backbone, you can stay closer to the manipulation numbers above.

---

## How TO/MPC data changes the data requirement

TO data is typically:

* high quality (good for sample efficiency)
* but narrow / low entropy (bad for coverage)

So you may need **more trajectories than you think** unless you deliberately add diversity.

A good target is:

* **~50–200 distinct initial conditions per task mode**
* **~10–50 perturbations/noise variants per initial condition**
* plus some explicit recovery data

This often ends up at:

* **hundreds to thousands of trajectories**, depending on (H)

---

## A concrete planning heuristic (works well)

### Minimum viable dataset (for state-based manipulation)

Aim for:

* **BC:** ( \ge 1\times 10^5 ) transitions
* **Offline RL:** ( \ge 5\times 10^5 ) transitions

### Comfortable dataset

* **BC:** ( \sim 5\times 10^5 ) transitions
* **Offline RL:** ( \sim 2\times 10^6 ) transitions

If your horizon is (H=200):

* (5\times 10^5) transitions ≈ 2,500 trajectories
* (2\times 10^6) transitions ≈ 10,000 trajectories

That’s totally plausible if TO is cheap in simulation.

---

## The real bottleneck isn’t raw size — it’s *coverage*

You can have 10 million transitions and still fail if:

* all trajectories follow the same narrow nominal path
* the policy never sees recovery states
* actions have almost no spread

### Better than “more data”:

* Randomize initial conditions
* Inject disturbances mid-trajectory and replan
* Add action noise then re-optimize (“noisy expert”)
* Include failure trajectories (especially for IQL/CQL)
* Domain randomize dynamics/contact parameters

These upgrades often reduce the needed dataset by **5–10×**.

---

## Quick diagnostics to know if you have “enough”

You likely have enough data if:

1. **BC validation loss stops improving** meaningfully
2. The policy succeeds on **held-out initial conditions**
3. State visitation of policy roughly matches dataset on-success cases
4. Offline RL improves returns without collapsing behavior distribution

If offline RL makes the policy worse, it’s usually a **coverage / OOD action** issue, not “too few gradient steps”.

---

## Bottom line

For **trajectory-optimization-generated state-based robotics data**, a solid rule of thumb is:

* **BC:** ~(10^5)–(10^6) transitions
* **Offline-safe RL:** ~(5\times10^5)–(5\times10^6) transitions
* (multiply by ~10×–100× if learning from pixels end-to-end)

---

If you tell me:

* your state dimension (or observation type),
* action dimension,
* typical horizon (H),
* number of tasks / goal conditions,

I can give you a tighter estimate and a concrete dataset-generation plan (how many initial states, perturbations, replans, and how to split train/val).
