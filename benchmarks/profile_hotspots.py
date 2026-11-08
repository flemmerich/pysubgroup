"""Profile DFS on synthetic data with 1,000 and 100,000 rows.

Prints the time per DFS node, the share of time spent in NumPy kernels, and the
functions with the highest self time.
Usage: ``uv run python benchmarks/profile_hotspots.py``.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from pysubgroup import (
    DFS,
    BinaryTarget,
    SubgroupDiscoveryTask,
    WRAccQF,
    count_calls,
    generate_conditions,
)

KERNELS = ("numpy", "bitwise", "count_nonzero", "reduce", "unpackbits", "bincount")


def make_data(rows: int, features: int = 12, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 5, size=(rows, features))
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(features)])
    y = ((values[:, 0] == 1) | ((values[:, 1] >= 3) & (values[:, 2] <= 1))) ^ (
        rng.random(rows) < 0.05
    )
    return X, y.astype(int)


def profile(label: str, function: Callable[[], object]) -> None:
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    function()
    profiler.disable()
    total = time.perf_counter() - started
    statistics = pstats.Stats(profiler)
    kernels = sum(
        own_time
        for (file, _, name), (_, _, own_time, _, _) in statistics.stats.items()
        if file == "~" and any(kernel in name for kernel in KERNELS)
    )
    print(f"{label}: {total:.2f}s (profiled), NumPy kernels {kernels / total:.0%}")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(6)
    print("\n".join(line for line in stream.getvalue().splitlines()
                    if line.strip()[:1].isdigit()))


def main() -> None:
    for rows in (1_000, 100_000):
        X, y = make_data(rows)
        data = X.assign(t=y)
        target = BinaryTarget().fit(data, "t", 1)
        space = generate_conditions(X)
        # Scalar and batched evaluations both count; DFS batches WRAcc by default.
        counted = count_calls(WRAccQF())
        task = SubgroupDiscoveryTask(data, target, space, counted, result_set_size=10, depth=3)
        started = time.perf_counter()
        DFS().execute(task)
        elapsed = time.perf_counter() - started
        subgroups = counted.call_counts.subgroups
        nodes = subgroups["evaluate"] + subgroups["evaluate_many"]
        print(f"\nDFS rows={rows}: {len(space)} conditions, {nodes} evaluated nodes, "
              f"{elapsed:.2f}s, {elapsed / max(nodes, 1) * 1e6:.0f} us/node")
        task = SubgroupDiscoveryTask(data, target, space, WRAccQF(), result_set_size=10, depth=3)
        profile(f"DFS rows={rows}", lambda task=task: DFS().execute(task))


if __name__ == "__main__":
    main()
