import unittest

import pysubgroup as ps
from tests.DataSets import get_credit_data


class TestUtilsDataConversion(unittest.TestCase):
    def setUp(self) -> None:
        data = get_credit_data()
        target = ps.BinaryTarget("class", b"bad")
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=["class"])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=["class"])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(
            data, target, searchSpace, result_set_size=2, depth=1, qf=ps.StandardQF(0.5)
        )
        self.result = ps.BeamSearch(1, beam_width_adaptive=True).execute(self.task)

    def test_to_dataframe(self):
        self.result.to_dataframe()

    def test_to_dataframe_inlude_target(self):
        self.result.to_dataframe(include_target=True)


if __name__ == "__main__":
    unittest.main()
