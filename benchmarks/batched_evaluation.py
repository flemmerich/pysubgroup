"""Runtime of scalar, batched (CPU), and GPU evaluation of subgroup discovery searches.

Synthetic data: categorical features with five values each, a binary label and a
numeric target that depend on two features. Every configuration runs each
evaluation mode once after a warm-up, and checks that all modes find the same
subgroups.

    uv run python benchmarks/batched_evaluation.py --rows 100000 --features 20
    uv run --extra gpu python benchmarks/batched_evaluation.py --rows 1000000 --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pysubgroup import (
    DFS,
    OPUS,
    BeamSearch,
    BestFirstSearch,
    BinaryTarget,
    NumericTarget,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WRAccQF,
    generate_conditions,
)
from pysubgroup.algorithms import base_algorithms as algorithms_module


def synthetic(rows: int, features: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {f"f{j}": rng.integers(0, 5, rows).astype(str) for j in range(features)}
    )
    signal = (data.f0 == "1") & (data.f1 != "2")
    data["label"] = (rng.random(rows) < 0.2 + 0.4 * signal).astype(int)
    data["value"] = rng.normal(size=rows) + signal
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="also run on the GPU with --device cuda")
    parser.add_argument("--skip-scalar", action="store_true",
                        help="skip the slow one-subgroup-at-a-time mode")
    args = parser.parse_args()

    data = synthetic(args.rows, args.features)
    conditions = generate_conditions(data.drop(columns=["label", "value"]))
    modes = [] if args.skip_scalar else [("scalar", {"batched": False})]
    modes.append(("batched", {"batched": True}))
    if args.device == "cuda":
        modes.append(("cuda", {"device": "cuda"}))
    measures = [
        ("WRAcc", BinaryTarget().fit(data, "label", 1), WRAccQF()),
        ("mean", NumericTarget().fit(data, "value"), StandardQFNumeric(0.5)),
    ]
    algorithms = [DFS(), BestFirstSearch(), OPUS(), BeamSearch()]
    # Search times exclude the setup of each search (input validation, encoding,
    # and the transfer to the device), which is the same in every mode.
    setup = [0.0]
    initialize = algorithms_module._Search.__init__

    def timed_initialize(self, *arguments, **keywords):
        start = time.perf_counter()
        initialize(self, *arguments, **keywords)
        setup[0] += time.perf_counter() - start

    algorithms_module._Search.__init__ = timed_initialize
    print(f"{args.rows} rows, {len(conditions)} conditions, depth {args.depth}; "
          "search time without setup")
    print(f"{'algorithm':<16}{'measure':<8}" + "".join(f"{name:>10}" for name, _ in modes))
    for label, target, qf in measures:
        for algorithm in algorithms:
            times, results = [], []
            for _, options in modes:
                task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=10,
                                             depth=args.depth, **options)
                if options.get("device") == "cuda":
                    algorithm.execute(task)  # warm-up: CUDA context and kernel compilation
                setup[0] = 0.0
                start = time.perf_counter()
                results.append(algorithm.execute(task))
                times.append(time.perf_counter() - start - setup[0])
            rules = [[rule for _, rule in result] for result in results]
            assert all(r == rules[0] for r in rules), f"modes differ for {algorithm}"
            print(f"{type(algorithm).__name__:<16}{label:<8}"
                  + "".join(f"{t:>9.3f}s" for t in times), flush=True)


if __name__ == "__main__":
    main()
