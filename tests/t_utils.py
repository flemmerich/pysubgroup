from collections import defaultdict



def assertResultEqual(self, result, s):
    alg_results = to_dict(result.to_descriptions())
    sol_results = to_dict(conjunctions_from_str(s))
    self.maxDiff=None
    self.assertEqual(len(alg_results), len(sol_results))
    #self.assertDictEqual(alg_results, sol_results)
    self.assertListEqual(sorted(alg_results.keys()), sorted(sol_results.keys()))
    for key in alg_results.keys():
        self.assertListEqual(alg_results[key], sol_results[key], msg=f"{key}")


def print_result(result):
    for _, (quality, pattern, _) in enumerate(result.to_descriptions(include_stats=True)):
        print(quality, pattern)

def conjunctions_from_str(s):
    import pysubgroup as ps # pylint:disable = import-outside-toplevel
    result = []
    for line in s.strip().splitlines():
        parts = line.strip().split(" ")
        q=float(parts[0])
        tmp = " ".join(parts[1:])
        result.append((q, ps.Conjunction.from_str(tmp)))
    return result

def interval_selectors_from_str(s):
    import pysubgroup as ps # pylint:disable = import-outside-toplevel
    result = []
    for line in s.strip().splitlines():
        result.append(ps.IntervalSelector.from_str(line))
    return result

def to_dict(result):
    out = defaultdict(list)
    for tpl in result:
        q = round(float(tpl[0]), 8)
        sg = tpl[1]
        out[q].append(sg)
    return {key: sorted(out[key]) for key in sorted(out.keys())}
