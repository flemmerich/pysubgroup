def test_algorithms_beamsearch_optimal_result():
    import numpy as np
    import pandas as pd

    import pysubgroup as ps

    data = pd.DataFrame(
        np.array(
            [
                #    a b c d   target
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        columns=["a", "b", "c", "d", "target"],
    )

    target = ps.BinaryTarget("target", 1)

    searchspace = [
        ps.EqualitySelector("a", 1),
        ps.EqualitySelector("b", 1),
        ps.EqualitySelector("c", 1),
        ps.EqualitySelector("d", 1),
    ]
    task = ps.SubgroupDiscoveryTask(
        data, target, searchspace, result_set_size=2, depth=2, qf=ps.StandardQF(a=0.5)
    )
    result = ps.BeamSearch(beam_width=3).execute(task)

    assert result.results[0][1] == ps.Conjunction([searchspace[2], searchspace[3]])


def test_algorithms_beamsearch_width_too_small():
    import numpy as np
    import pandas as pd

    import pysubgroup as ps

    data = pd.DataFrame(
        np.array(
            [
                #    a b c d   target
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        columns=["a", "b", "c", "d", "target"],
    )

    target = ps.BinaryTarget("target", 1)

    searchspace = [
        ps.EqualitySelector("a", 1),
        ps.EqualitySelector("b", 1),
        ps.EqualitySelector("c", 1),
        ps.EqualitySelector("d", 1),
    ]
    task = ps.SubgroupDiscoveryTask(
        data, target, searchspace, result_set_size=2, depth=2, qf=ps.StandardQF(a=0.5)
    )
    result = ps.BeamSearch(beam_width=2).execute(task)

    assert set([ps.Conjunction(s) for s in searchspace[:2]]) == set(
        [sg for _, sg, _ in result.results]
    )
