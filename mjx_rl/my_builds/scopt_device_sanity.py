#!/usr/bin/env python3
"""Sanity check JAX device placement and timings for SCOPT helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import jax
import jax.numpy as jnp

try:
    from mjx_rl.my_builds.scopt import (
        load_franka_scopt_model,
        make_jitted_dynamics,
        make_jitted_state_space_jacobians,
    )
except ModuleNotFoundError:
    # Allow running this file directly without pre-setting PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, repo_root.as_posix())
    from mjx_rl.my_builds.scopt import (
        load_franka_scopt_model,
        make_jitted_dynamics,
        make_jitted_state_space_jacobians,
    )


METHODS = ("euler", "zoh", "rk4_linearize")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check JAX device placement + timings for scopt.py helpers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "gpu", "tpu"),
        help="Target accelerator class.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=METHODS,
        help="Discretization methods to benchmark for state-space Jacobians.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Timestep used for discrete Jacobian benchmarking.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup calls before timing.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed iterations per benchmark.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Optional explicit path to Franka XML.",
    )
    parser.add_argument(
        "--enable-collisions",
        action="store_true",
        help="Enable collision geoms (default is disabled).",
    )
    return parser.parse_args()


def _select_device(device_kind: str) -> jax.Device:
    if device_kind == "auto":
        return jax.devices()[0]
    devs = jax.devices(device_kind)
    if not devs:
        available = ", ".join(str(d) for d in jax.devices())
        raise RuntimeError(
            f"No JAX devices found for '{device_kind}'. Available: {available}"
        )
    return devs[0]


def _block_tree(tree: Any) -> Any:
    def _block_leaf(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_block_leaf, tree)


def _tree_devices(tree: Any) -> list[str]:
    devices: set[str] = set()

    def _collect(x: Any) -> Any:
        if hasattr(x, "devices"):
            try:
                for dev in x.devices():
                    devices.add(str(dev))
            except TypeError:
                pass
        elif hasattr(x, "device"):
            dev = x.device
            devices.add(str(dev() if callable(dev) else dev))
        return x

    jax.tree_util.tree_map(_collect, tree)
    return sorted(devices)


def _all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    checks = [bool(jnp.all(jnp.isfinite(x))) for x in leaves if hasattr(x, "shape")]
    return all(checks) if checks else True


def _shape_summary(tree: Any) -> list[tuple[int, ...]]:
    leaves = jax.tree_util.tree_leaves(tree)
    return [tuple(x.shape) for x in leaves if hasattr(x, "shape")]


def _bench(fn: Any, args: Iterable[Any], warmup: int, iters: int) -> tuple[Any, float]:
    for _ in range(max(warmup, 0)):
        _block_tree(fn(*args))

    t0 = time.perf_counter()
    out = None
    for _ in range(max(iters, 1)):
        out = fn(*args)
        _block_tree(out)
    elapsed_s = time.perf_counter() - t0
    ms_per_iter = 1e3 * elapsed_s / max(iters, 1)
    return out, ms_per_iter


def main() -> None:
    args = _parse_args()
    target_device = _select_device(args.device)

    print(f"JAX backend: {jax.default_backend()}")
    print("Available devices:")
    for dev in jax.devices():
        print(f"  - {dev}")
    print(f"Target device: {target_device}")
    print(f"Methods: {', '.join(args.methods)}")

    with jax.default_device(target_device):
        model = load_franka_scopt_model(
            xml_path=args.xml_path,
            disable_collisions=not args.enable_collisions,
        )
        q = jnp.array(
            [0.9207, 0.2574, -0.9527, -2.0683, 0.2799, 2.1147, 2.0], dtype=jnp.float32
        )
        qdot = jnp.zeros((7,), dtype=jnp.float32)
        tau = jnp.zeros((7,), dtype=jnp.float32)
        dt = jnp.array(args.dt, dtype=jnp.float32)

        # Explicit device placement for clarity in reports.
        q = jax.device_put(q, target_device)
        qdot = jax.device_put(qdot, target_device)
        tau = jax.device_put(tau, target_device)
        dt = jax.device_put(dt, target_device)

        qdd_fn, jac_fn = make_jitted_dynamics(model)

        qdd_out, qdd_ms = _bench(qdd_fn, (q, qdot, tau), args.warmup, args.iters)
        jac_out, jac_ms = _bench(jac_fn, (q, qdot, tau), args.warmup, args.iters)

        print("\nqddot:")
        print(f"  shapes: {_shape_summary(qdd_out)}")
        print(f"  finite: {_all_finite(qdd_out)}")
        print(f"  devices: {', '.join(_tree_devices(qdd_out))}")
        print(f"  avg latency: {qdd_ms:.4f} ms/iter")

        print("\ndynamics jacobians:")
        print(f"  shapes: {_shape_summary(jac_out)}")
        print(f"  finite: {_all_finite(jac_out)}")
        print(f"  devices: {', '.join(_tree_devices(jac_out))}")
        print(f"  avg latency: {jac_ms:.4f} ms/iter")

        for method in args.methods:
            cont_fn, disc_fn = make_jitted_state_space_jacobians(model, method=method)
            cont_out, cont_ms = _bench(cont_fn, (q, qdot, tau), args.warmup, args.iters)
            disc_out, disc_ms = _bench(
                disc_fn, (q, qdot, tau, dt), args.warmup, args.iters
            )

            print(f"\nstate-space jacobians ({method}):")
            print(f"  continuous shapes: {_shape_summary(cont_out)}")
            print(f"  continuous finite: {_all_finite(cont_out)}")
            print(f"  continuous devices: {', '.join(_tree_devices(cont_out))}")
            print(f"  continuous avg latency: {cont_ms:.4f} ms/iter")
            print(f"  discrete shapes: {_shape_summary(disc_out)}")
            print(f"  discrete finite: {_all_finite(disc_out)}")
            print(f"  discrete devices: {', '.join(_tree_devices(disc_out))}")
            print(f"  discrete avg latency: {disc_ms:.4f} ms/iter")


if __name__ == "__main__":
    main()
