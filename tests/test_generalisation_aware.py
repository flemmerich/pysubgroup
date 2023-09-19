import unittest

import numpy as np
import pandas as pd
from algorithms_testing import TestAlgorithmsBase
from t_utils import conjunctions_from_str

import pysubgroup as ps
from pysubgroup.datasets import get_credit_data


class TestGeneralisationAwareQf(unittest.TestCase):
    def setUp(self):
        self.qf = ps.CountQF()
        self.ga_qf = ps.GeneralizationAwareQF(self.qf)
        self.prepare_df()

    def prepare_df(self):
        A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        self.A1 = ps.EqualitySelector("columnA", True)
        self.A0 = ps.EqualitySelector("columnA", False)

        B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        self.BA = ps.EqualitySelector("columnB", "A")
        self.BC = ps.EqualitySelector("columnB", "C")
        self.BD = ps.EqualitySelector("columnB", "D")
        self.df = pd.DataFrame.from_dict(
            {
                "columnA": A,
                "columnB": B,
                "columnC": np.array([[0, 1] for _ in range(5)]).flatten(),
            }
        )

    def test_CountTarget1(self):
        df = self.df
        target = ps.FITarget()
        self.ga_qf.calculate_constant_statistics(df, target)

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1]), target, df)

        A1_score = self.qf.evaluate(ps.Conjunction([self.A1]), target, df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), target, df)

        self.assertEqual(ga_score, A1_score - zero_score)

        ga2_score = self.ga_qf.evaluate(ps.Conjunction([self.A1]), target, df)

        self.assertEqual(ga2_score, ga_score)

    def test_CountTarget2(self):
        df = self.df
        target = ps.FITarget()
        self.ga_qf.calculate_constant_statistics(df, None)

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1, self.BA]), target, df)

        A_B_score = self.qf.evaluate(ps.Conjunction([self.A1, self.BA]), target, df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), target, df)

        self.assertEqual(ga_score, A_B_score - zero_score)


class TestGeneralisationAware_StandardQf(unittest.TestCase):
    def setUp(self):
        self.df = None
        self.A1 = None
        self.BA = None
        self.A0 = None
        self.BD = None
        TestGeneralisationAwareQf.prepare_df(self)
        self.ga_qf = ps.GeneralizationAware_StandardQF(
            0, optimistic_estimate_strategy="max"
        )

    def test_simple(self):
        target = ps.BinaryTarget("columnC", 1)
        qf = ps.StandardQF(0)
        qf.calculate_constant_statistics(self.df, target)

        self.ga_qf.calculate_constant_statistics(self.df, target)

        # print(qf.calculate_statistics(self.A1, self.df))
        # print(qf.calculate_statistics(self.BA, self.df))
        # print(qf.calculate_statistics(ps.Conjunction([self.A1, self.BA]), self.df))
        # print(qf.calculate_statistics(slice(None), self.df))
        ga_stat = self.ga_qf.calculate_statistics(
            ps.Conjunction([self.A1, self.BA]), target, self.df
        )

        self.assertEqual(ga_stat.subgroup_stats, ps.SimplePositivesQF.tpl(3, 2))
        self.assertEqual(ga_stat.generalisation_stats, ps.SimplePositivesQF.tpl(5, 3))

        # Ensure cache works properly
        self.assertIs(
            ga_stat,
            self.ga_qf.calculate_statistics(
                ps.Conjunction([self.A1, self.BA]), target, self.df
            ),
        )

        ga_score = self.ga_qf.evaluate(
            ps.Conjunction([self.A1, self.BA]), target, self.df
        )
        ga_score2 = self.ga_qf.evaluate(
            ps.Conjunction([self.A1, self.BA]), target, self.df
        )
        ga_score3 = self.ga_qf.evaluate(
            ps.Conjunction([self.A0, self.BD]), target, self.df
        )
        self.assertEqual(ga_score, ga_score2)
        self.assertAlmostEqual(ga_score, 0.06666666666666)
        self.assertTrue(np.isnan(ga_score3))

    def test_error(self):
        with self.assertRaises(ValueError):
            ps.GeneralizationAware_StandardQF(0.5, "blabla")


class TestSimpleGA(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        conj_list = conjunctions_from_str(
            """0.05280000000000001 checking_status=='b'<0''
   0.03610000000000002 savings_status=='b'<100''
   0.0243      checking_status=='b'0<=X<200''
   0.0208      property_magnitude=='b'no known property''
   0.0188      purpose=='b'new car''
   0.0184      employment=='b'<1''
   0.0163      housing=='b'rent''
   0.016000000000000007        personal_status=='b'female div/dep/mar''
   0.015300000000000003        other_payment_plans=='b'bank''
   0.0133      credit_history=='b'all paid''"""
        )
        self.result = [conjunction for quality, conjunction in conj_list]
        self.qualities = [quality for quality, conjunction in conj_list]
        data = get_credit_data()
        target = ps.BinaryTarget("class", b"bad")
        searchSpace = ps.create_nominal_selectors(data, ignore=["class"])
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=3,
            qf=ps.GeneralizationAwareQF(ps.StandardQF(1.0)),
        )

    def test_GA_SimpleDFS(self):
        self.runAlgorithm(
            ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task
        )


class TestGeneralizationAware_StandardQF_a05(TestAlgorithmsBase, unittest.TestCase):
    def get_a(self):
        return 0.5

    def get_output_str(self):
        return """0.10086921502691486 checking_status=='b'<0''
        0.065       credit_history=='b'no credits/all paid''
        0.0600832755431992  credit_history=='b'all paid''
        0.05300330790951305 property_magnitude=='b'no known property''
        0.046852215652241715        checking_status=='b'0<=X<200''
        0.046488822458723086        savings_status=='b'<100''
        0.04436633963967792 employment=='b'<1''
        0.043698221718866   other_payment_plans=='b'bank'' AND purpose=='b'new car''
        0.04183300132670378 housing=='b'rent'' AND property_magnitude=='b'no known property'' AND savings_status=='b'<100''
        0.04103779623011525 other_payment_plans=='b'bank''"""  # noqa: 501

    def setUp(self):
        conj_list = conjunctions_from_str(self.get_output_str())
        self.result = [conjunction for quality, conjunction in conj_list]
        self.qualities = [quality for quality, conjunction in conj_list]

        data = get_credit_data()
        target = ps.BinaryTarget("class", b"bad")
        searchSpace = ps.create_nominal_selectors(data, ignore=["class"])
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=3,
            qf=ps.GeneralizationAware_StandardQF(self.get_a()),
        )

    def test_SimpleDFS(self):
        self.task.qf = ps.GeneralizationAware_StandardQF(self.get_a())
        self.runAlgorithm(
            ps.SimpleDFS(),
            f"StandardQF_SimpleDFS, a={self.get_a()}",
            self.result,
            self.qualities,
            self.task,
        )

    def test_Apriori_diff(self):
        self.task.qf = ps.GeneralizationAware_StandardQF(
            self.get_a(), optimistic_estimate_strategy="difference"
        )
        apriori = ps.Apriori()
        apriori.use_vectorization = False
        self.runAlgorithm(
            apriori,
            f"StandardQF_Apriori diff, a={self.get_a()}",
            self.result,
            self.qualities,
            self.task,
        )

    def test_Apriori_max(self):
        self.task.qf = ps.GeneralizationAware_StandardQF(
            self.get_a(), optimistic_estimate_strategy="max"
        )
        apriori = ps.Apriori()
        apriori.use_vectorization = False
        self.runAlgorithm(
            apriori,
            f"StandardQF_Apriori, max, a={self.get_a()}",
            self.result,
            self.qualities,
            self.task,
        )


class TestGeneralizationAware_StandardQF_a(TestGeneralizationAware_StandardQF_a05):
    def get_a(self):
        return 1

    def get_output_str(self):
        return """   0.05280000000000001 checking_status=='b'<0''
   0.03610000000000002 savings_status=='b'<100''
   0.0243      checking_status=='b'0<=X<200''
   0.0208      property_magnitude=='b'no known property''
   0.0188      purpose=='b'new car''
   0.0184      employment=='b'<1''
   0.0163      housing=='b'rent''
   0.016000000000000007        personal_status=='b'female div/dep/mar''
   0.015300000000000003        other_payment_plans=='b'bank''
   0.0133      credit_history=='b'all paid''"""


class TestGeneralizationAware_StandardQF_a0_d2(TestGeneralizationAware_StandardQF_a05):
    def get_a(self):
        return 0

    def get_output_str(self):
        return """    0.6795580110497237  job=='b'unemp/unskilled non res'' AND purpose=='b'furniture/equipment''
            0.6666666666666667  purpose=='b'domestic appliance'' AND savings_status=='b'100<=X<500''
            0.6666666666666667  personal_status=='b'male mar/wid'' AND purpose=='b'domestic appliance''
            0.6666666666666667  job=='b'unskilled resident'' AND purpose=='b'domestic appliance''
   0.6554054054054055  foreign_worker=='b'no'' AND job=='b'high qualif/self emp/mgmt''
   0.6363636363636364  purpose=='b'repairs'' AND savings_status=='b'>=1000''
   0.6290322580645161  employment=='b'unemployed'' AND purpose=='b'domestic appliance''
   0.6089385474860336  housing=='b'rent'' AND purpose=='b'retraining''
   0.6 foreign_worker=='b'no'' AND personal_status=='b'male div/sep''
   0.5957446808510638  other_payment_plans=='b'stores'' AND purpose=='b'repairs''"""  # noqa: 501

    def setUp(self):
        conj_list = conjunctions_from_str(self.get_output_str())
        self.result = [conjunction for quality, conjunction in conj_list]
        self.qualities = [quality for quality, conjunction in conj_list]

        data = get_credit_data()
        target = ps.BinaryTarget("class", b"bad")
        searchSpace = ps.create_nominal_selectors(data, ignore=["class"])
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=2,
            qf=ps.GeneralizationAware_StandardQF(self.get_a()),
        )


class TestGeneralizationAware_StandardQF_a0(TestGeneralizationAware_StandardQF_a05):
    def get_a(self):
        return 0

    def get_output_str(self):
        return """   0.7 job=='b'unskilled resident'' AND own_telephone=='b'yes'' AND personal_status=='b'male mar/wid''
   0.7 foreign_worker=='b'no'' AND other_parties=='b'guarantor'' AND personal_status=='b'male mar/wid''
   0.7 employment=='b'>=7'' AND job=='b'unskilled resident'' AND purpose=='b'used car''
   0.7 credit_history=='b'critical/other existing credit'' AND job=='b'unskilled resident'' AND purpose=='b'used car''
   0.7 checking_status=='b'>=200'' AND personal_status=='b'male mar/wid'' AND savings_status=='b'500<=X<1000''
   0.7 checking_status=='b'>=200'' AND own_telephone=='b'yes'' AND personal_status=='b'male mar/wid''
   0.7 checking_status=='b'>=200'' AND credit_history=='b'critical/other existing credit'' AND personal_status=='b'male mar/wid''
   0.6939655172413793  other_parties=='b'guarantor'' AND property_magnitude=='b'life insurance'' AND savings_status=='b'no known savings''
   0.6939655172413793  foreign_worker=='b'no'' AND other_parties=='b'guarantor'' AND property_magnitude=='b'life insurance''
   0.6818181818181819  credit_history=='b'delayed previously'' AND property_magnitude=='b'real estate'' AND savings_status=='b'500<=X<1000''"""  # noqa: 501

    def setUp(self):
        conj_list = conjunctions_from_str(self.get_output_str())
        self.result = [conjunction for quality, conjunction in conj_list]
        self.qualities = [quality for quality, conjunction in conj_list]

        data = get_credit_data()
        target = ps.BinaryTarget("class", b"bad")
        searchSpace = ps.create_nominal_selectors(data, ignore=["class"])
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=3,
            qf=ps.GeneralizationAware_StandardQF(self.get_a()),
        )


class TestGeneralizationAware_StandardQFNumeric(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        conj_list = conjunctions_from_str(
            """   832.5979220717699   job=='b'high qualif/self emp/mgmt''
   673.6338022041458   purpose=='b'used car''
   645.953015714855    property_magnitude=='b'no known property''
   603.3209078187183   own_telephone=='b'yes''
   576.235405327832    class=='b'bad'' AND own_telephone=='b'yes''
   540.9390501453018   purpose=='b'other''
   537.3010282319029   housing=='b'for free''
   440.3787869550485   checking_status=='b'0<=X<200'' AND foreign_worker=='b'no'' AND property_magnitude=='b'life insurance''
   407.6428886169854   checking_status=='b'0<=X<200'' AND foreign_worker=='b'no'' AND other_payment_plans=='b'bank''
   406.8834000000001   credit_history=='b'no credits/all paid'' """  # noqa: 501
        )
        self.result = [conjunction for quality, conjunction in conj_list]
        self.qualities = [quality for quality, conjunction in conj_list]

        data = get_credit_data()
        target = ps.NumericTarget("credit_amount")
        searchSpace_Nominal = ps.create_nominal_selectors(
            data, ignore=["credit_amount"]
        )
        searchSpace_Numeric = (
            []
        )  # ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=3,
            qf=ps.GeneralizationAware_StandardQFNumeric(1, False, centroid="mean"),
        )

    def test_SimpleDFS(self):
        self.task.qf = ps.GeneralizationAware_StandardQFNumeric(0.5)
        self.runAlgorithm(
            ps.SimpleDFS(),
            "Numeric StandardQF_SimpleDFS",
            self.result,
            self.qualities,
            self.task,
        )

    # def test_DFS(self):
    #     self.task.qf = ps.GeneralizationAware_StandardQFNumeric(0.5)
    #     apriori = ps.Apriori()
    #     apriori.use_vectorization = False
    #     self.runAlgorithm(
    #         apriori, "StandardQF_Apriori", self.result, self.qualities, self.task
    #     )


if __name__ == "__main__":
    unittest.main(module="test_generalisation_aware")
