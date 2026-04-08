"""Franka MJX dynamics helpers for SCOPT-style linearization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from etils import epath
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import mujoco
from mujoco import mjx


@dataclass(frozen=True)
class FrankaSCOPTModel:
    """Container for MuJoCo/MJX model handles and useful dimensions."""

    mj_model: mujoco.MjModel
    mjx_model: Any
    nq: int
    nu: int
    u_low: jax.Array
    u_high: jax.Array


def _resolve_franka_xml(xml_path: str | Path | None = None) -> epath.Path:
    if xml_path is not None:
        return epath.Path(xml_path)

    default_rel = epath.Path("mujoco_menagerie/franka_emika_panda/panda_nohand.xml")
    if default_rel.exists():
        return default_rel

    repo_root_fallback = epath.Path(__file__).resolve().parents[2] / default_rel
    return repo_root_fallback


def load_franka_scopt_model(
    xml_path: str | Path | None = None,
    disable_collisions: bool = True,
    timestep: float | None = None,
) -> FrankaSCOPTModel:
    """Load Franka arm model and configure direct torque control."""

    model_path = _resolve_franka_xml(xml_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Franka XML not found at: {model_path}")

    mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())

    if disable_collisions:
        # Menagerie Panda collision geoms are in group 3.
        collision_mask = mj_model.geom_group == 3
        mj_model.geom_contype[collision_mask] = 0
        mj_model.geom_conaffinity[collision_mask] = 0

    # Match the MJX env setup for arm-only dynamics.
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    if timestep is not None:
        mj_model.opt.timestep = float(timestep)

    if mj_model.nu != 7:
        raise ValueError(
            f"Expected 7 arm actuators in panda_nohand.xml, got nu={mj_model.nu}."
        )

    arm = slice(0, 7)
    mj_model.actuator_gainprm[arm, 0] = 1.0
    mj_model.actuator_biasprm[arm, :] = 0.0
    mj_model.actuator_ctrlrange[:4, 0] = -81.0
    mj_model.actuator_ctrlrange[:4, 1] = 81.0
    mj_model.actuator_ctrlrange[4:7, 0] = -12.0
    mj_model.actuator_ctrlrange[4:7, 1] = 12.0

    mjx_model = mjx.put_model(mj_model)
    u_bounds = jnp.asarray(mj_model.actuator_ctrlrange, dtype=jnp.float32)
    return FrankaSCOPTModel(
        mj_model=mj_model,
        mjx_model=mjx_model,
        nq=int(mj_model.nq),
        nu=int(mj_model.nu),
        u_low=u_bounds[:, 0],
        u_high=u_bounds[:, 1],
    )


def _validate_shapes(
    model: FrankaSCOPTModel,
    q: jax.Array,
    qdot: jax.Array,
    tau: jax.Array,
) -> None:
    if q.shape != (model.nq,):
        raise ValueError(f"Expected q shape {(model.nq,)}, got {q.shape}.")
    if qdot.shape != (model.nq,):
        raise ValueError(f"Expected qdot shape {(model.nq,)}, got {qdot.shape}.")
    if tau.shape != (model.nu,):
        raise ValueError(f"Expected tau shape {(model.nu,)}, got {tau.shape}.")


def franka_qddot(
    model: FrankaSCOPTModel,
    q: jax.Array,
    qdot: jax.Array,
    tau: jax.Array,
) -> jax.Array:
    """Continuous-time joint acceleration f(q, qdot, tau) = qddot."""

    q = jnp.asarray(q, dtype=jnp.float32)
    qdot = jnp.asarray(qdot, dtype=jnp.float32)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    _validate_shapes(model, q, qdot, tau)

    data = mjx.make_data(model.mjx_model)
    data = data.replace(
        qpos=q,
        qvel=qdot,
        ctrl=jnp.clip(tau, model.u_low, model.u_high),
    )
    data = mjx.forward(model.mjx_model, data)
    return data.qacc.astype(jnp.float32)


def franka_dynamics_jacobians(
    model: FrankaSCOPTModel,
    q: jax.Array,
    qdot: jax.Array,
    tau: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return dqddot/dq, dqddot/dqdot, dqddot/dtau."""

    q = jnp.asarray(q, dtype=jnp.float32)
    qdot = jnp.asarray(qdot, dtype=jnp.float32)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    _validate_shapes(model, q, qdot, tau)

    dynamics_fn = lambda _q, _qdot, _tau: franka_qddot(model, _q, _qdot, _tau)
    dq, dqdot, dtau = jax.jacfwd(dynamics_fn, argnums=(0, 1, 2))(q, qdot, tau)
    return dq.astype(jnp.float32), dqdot.astype(jnp.float32), dtau.astype(jnp.float32)


def franka_state_space_jacobians(
    model: FrankaSCOPTModel,
    q: jax.Array,
    qdot: jax.Array,
    tau: jax.Array,
    dt: float | None = None,
    method: Literal["euler", "zoh", "rk4_linearize"] = "euler",
) -> tuple[jax.Array, jax.Array]:
    """Linearize x=[q; qdot], xdot=[qdot; qddot] into A, B.

    If `dt is None`, returns continuous-time Jacobians (A_c, B_c).
    If `dt` is given, returns discrete-time (A_d, B_d) using:
      - "euler": linearize-first, first-order Euler
      - "zoh": linearize-first, exact ZOH via block matrix exponential
      - "rk4_linearize": discretize-first with RK4, then linearize map
    """

    dq, dqdot, dtau = franka_dynamics_jacobians(model, q, qdot, tau)

    nq = model.nq
    nu = model.nu
    z_qq = jnp.zeros((nq, nq), dtype=jnp.float32)
    i_q = jnp.eye(nq, dtype=jnp.float32)
    z_qu = jnp.zeros((nq, nu), dtype=jnp.float32)

    a_top = jnp.concatenate([z_qq, i_q], axis=1)
    a_bottom = jnp.concatenate([dq, dqdot], axis=1)
    a_cont = jnp.concatenate([a_top, a_bottom], axis=0)
    b_cont = jnp.concatenate([z_qu, dtau], axis=0)

    if dt is None:
        return a_cont, b_cont

    dt = jnp.asarray(dt, dtype=jnp.float32)
    method = method.lower()

    if method == "euler":
        i_x = jnp.eye(2 * nq, dtype=jnp.float32)
        a_disc = i_x + dt * a_cont
        b_disc = dt * b_cont
        return a_disc, b_disc

    if method == "zoh":
        # Exact discrete-time map for linearized continuous system with held input.
        nx = 2 * nq
        nu = model.nu
        aug = jnp.zeros((nx + nu, nx + nu), dtype=jnp.float32)
        aug = aug.at[:nx, :nx].set(a_cont)
        aug = aug.at[:nx, nx:].set(b_cont)
        md = jsp_linalg.expm(dt * aug)
        a_disc = md[:nx, :nx]
        b_disc = md[:nx, nx:]
        return a_disc, b_disc

    if method == "rk4_linearize":
        _validate_shapes(model, q, qdot, tau)
        x0 = jnp.concatenate([q, qdot], axis=0).astype(jnp.float32)
        u0 = jnp.asarray(tau, dtype=jnp.float32)

        def _xdot(x: jax.Array, u: jax.Array) -> jax.Array:
            qx = x[:nq]
            vx = x[nq:]
            qdd = franka_qddot(model, qx, vx, u)
            return jnp.concatenate([vx, qdd], axis=0).astype(jnp.float32)

        def _rk4_step(x: jax.Array, u: jax.Array) -> jax.Array:
            k1 = _xdot(x, u)
            k2 = _xdot(x + 0.5 * dt * k1, u)
            k3 = _xdot(x + 0.5 * dt * k2, u)
            k4 = _xdot(x + dt * k3, u)
            return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        a_disc, b_disc = jax.jacfwd(_rk4_step, argnums=(0, 1))(x0, u0)
        return a_disc.astype(jnp.float32), b_disc.astype(jnp.float32)

    raise ValueError(
        f"Unknown method='{method}'. Use one of: 'euler', 'zoh', 'rk4_linearize'."
    )


def make_jitted_dynamics(
    model: FrankaSCOPTModel,
) -> tuple[Any, Any]:
    """Create JIT-compiled qddot and jacobian callables."""

    dynamics_fn = lambda q, qdot, tau: franka_qddot(model, q, qdot, tau)
    jac_fn = jax.jacfwd(dynamics_fn, argnums=(0, 1, 2))
    return jax.jit(dynamics_fn), jax.jit(jac_fn)


def make_jitted_state_space_jacobians(
    model: FrankaSCOPTModel,
    method: Literal["euler", "zoh", "rk4_linearize"] = "euler",
) -> tuple[Any, Any]:
    """Create JIT-compiled continuous/discrete state-space Jacobian callables.

    Returns:
      - cont_fn(q, qdot, tau) -> (A_c, B_c)
      - disc_fn(q, qdot, tau, dt) -> (A_d, B_d)
    """

    method = method.lower()
    if method not in ("euler", "zoh", "rk4_linearize"):
        raise ValueError(
            f"Unknown method='{method}'. Use one of: 'euler', 'zoh', 'rk4_linearize'."
        )

    cont_fn = lambda q, qdot, tau: franka_state_space_jacobians(
        model, q, qdot, tau, dt=None, method=method
    )
    disc_fn = lambda q, qdot, tau, dt: franka_state_space_jacobians(
        model, q, qdot, tau, dt=dt, method=method
    )
    return jax.jit(cont_fn), jax.jit(disc_fn)
