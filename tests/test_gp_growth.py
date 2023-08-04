import unittest

import pandas as pd
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.t_utils import assertResultEqual



class TestGpGrowth(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records([(1,1,0), (1,1,1,), (1,1,0), (1,0,1)], columns=("A", "B", "C"))

        target = ps.FITarget()
        QF=ps.CountQF()

        self.df1 = pd.DataFrame.from_records([(1,1,1,0), (1,1,0,1), (1,1,1,0), (1,1,0,1),(1,1,0,0), (1,0,0,0)], columns=("A", "B", "C", "D"))
        selectors1 = [ps.EqualitySelector(x, 1) for x in self.df1.columns]
        self.task1 = ps.SubgroupDiscoveryTask(self.df1, target, selectors1, result_set_size=20, depth=4, qf=QF)
        self.solution1 = """6 A==1
        6 Dataset
        5 B==1
        5 A==1 AND B==1
        2 D==1
        2 A==1 AND D==1
        2 A==1 AND B==1 AND D==1
        2 A==1 AND B==1 AND C==1
        2 B==1 AND D==1
        2 A==1 AND C==1
        2 B==1 AND C==1
        2 C==1"""


        self.df2 = pd.DataFrame.from_records([(1,1,1,0), (1,1,0,1, 1), (1,1,1,0), (1,1,0,1, 1),], columns=("A", "B", "C", "D", "E"))
        selectors2 = [ps.EqualitySelector(x, 1) for x in self.df2.columns]
        self.task2 = ps.SubgroupDiscoveryTask(self.df2, target, selectors2, result_set_size=20, depth=4, qf=QF)


        self.df3 = pd.DataFrame.from_records([(1,1,), (0,1), (1,0), (1,0)], columns=("A", "B"))
        selectors3 = [ps.EqualitySelector(x, 1) for x in self.df3.columns]
        self.task3 = ps.SubgroupDiscoveryTask(self.df3, target, selectors3, result_set_size=20, depth=4, qf=QF)
        self.all_selectors = {x: ps.EqualitySelector(x, 1) for x in self.df2.columns}


        self.task4 = ps.SubgroupDiscoveryTask(self.df1, target, selectors1, result_set_size=20, depth=4, qf=QF, constraints=[ps.MinSupportConstraint(3)])
        self.solution4 = """6 A==1
        6 Dataset
        5 B==1
        5 A==1 AND B==1"""

        self.task5 = ps.SubgroupDiscoveryTask(self.df1, target, selectors1, result_set_size=20, depth=4, qf=QF, constraints=[ps.MinSupportConstraint(10)])
        self.solution5 = """"""





    def test_export_fi(self):
        target = ps.FITarget()
        searchspace = ps.create_selectors(self.df1)
        task = ps.SubgroupDiscoveryTask(self.df1, target, searchspace, result_set_size=5, depth=2, qf=ps.CountQF())
        ps.GpGrowth().to_file(task, "./test_gp_fi.txt")

    def test_export_binary(self):
        target = ps.BinaryTarget("A", 1)
        searchspace = ps.create_selectors(self.df, ignore=["A"])
        task = ps.SubgroupDiscoveryTask(self.df, target, searchspace, result_set_size=5, depth=2, qf=ps.StandardQF(0.5))
        ps.GpGrowth().to_file(task, "./test_gp_binary.txt")

    def test_export_model(self):
        model = ps.PolyRegression_ModelClass("A", "B")
        QF = ps.EMM_Likelihood(model)
        searchspace = ps.create_selectors(self.df)
        task = ps.SubgroupDiscoveryTask(self.df, None, searchspace, result_set_size=5, depth=2, qf=QF)
        ps.GpGrowth().to_file(task, "./test_gp_model.txt")

    def test_gp_modes_restricted(self):
        with self.assertRaises(AssertionError):
            ps.GpGrowth(mode='fail').execute(self.task1)

    def test_gp_qf_requires_cov(self):
        tmp = self.task1.qf.gp_requires_cover_arr
        #del self.task1.qf.gp_requires_cover_arr
        self.task1.qf.gp_requires_cover_arr = True
        with self.assertWarns(UserWarning):
            result = ps.GpGrowth(mode='t_d').execute(self.task1)
        self.task1.qf.gp_requires_cover_arr = tmp
        assertResultEqual(self, result,self.solution1)

    def test_gp_b_u_simple1(self):
        result = ps.GpGrowth(mode='b_u').execute(self.task1)
        #print_result(result)
        assertResultEqual(self, result, self.solution1)

    def test_gp_simple1(self):
        with self.assertWarns(UserWarning):
            result = ps.GpGrowth(mode='t_d').execute(self.task1)
        #print_result(result)
        assertResultEqual(self, result,self.solution1)

        #self.assertEqual(results[0], (6, ps.Conjunction([self.all_selectors["A"]]) ))


    def test_gp_simple2(self):
        with self.assertWarns(UserWarning):
            result = ps.GpGrowth(mode='t_d').execute(self.task2)
        #print_result(result)
        assertResultEqual(self, result,
        """ 4 A==1 AND B==1
            4 A==1
            4 Dataset
            4 B==1
            2 A==1 AND B==1 AND E==1
            2 A==1 AND B==1 AND D==1 AND E==1
            2 B==1 AND E==1
            2 A==1 AND B==1 AND D==1
            2 A==1 AND E==1
            2 A==1 AND D==1 AND E==1
            2 B==1 AND D==1
            2 B==1 AND D==1 AND E==1
            2 A==1 AND B==1 AND D==1
            2 C==1
            2 A==1 AND C==1
            2 A==1 AND D==1
            2 A==1 AND D==1
            2 B==1 AND C==1
            2 B==1 AND D==1
            2 A==1 AND B==1 AND C==1""")


    def test_gp_simple3(self):
        with self.assertWarns(UserWarning):
            result = ps.GpGrowth(mode='t_d').execute(self.task3)
        #print_result(result)
        assertResultEqual(self, result,
        """4 Dataset
            3 A==1
            2 B==1
            1 A==1 AND B==1
            1 B==1""")


    def test_gp_simple4(self):
        result = ps.GpGrowth(mode='t_d').execute(self.task4)
        #print_result(result)
        assertResultEqual(self, result,self.solution4)


    def test_gp_simple5(self):
        result = ps.GpGrowth(mode='t_d').execute(self.task5)
        #print_result(result)
        assertResultEqual(self, result,self.solution5)

    def test_gp_credit(self):
        data = get_credit_data()
        #warnings.filterwarnings("error")
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['duration', 'credit_amount', 'class'])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['duration', 'credit_amount', 'class'])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        #searchSpace = [ps.EqualitySelector("checking_status", b'<0'), ps.EqualitySelector("own_telephone", b'yes'), ps.IntervalSelector("age", 30, 36)]
        #QF=model_target.EMM_Likelihood(model_target.PolyRegression_ModelClass(x_name='duration', y_name='credit_amount'))
        #target = ps.FITarget()
        #QF=ps.CountQF()

        QF=ps.StandardQF(0.5)
        target = ps.BinaryTarget('class', b'bad')
        task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=30, depth=2, qf=QF, constraints=[ps.MinSupportConstraint(30)])
        result = ps.GpGrowth(mode='t_d').execute(task)
        #print_result(result)
        assertResultEqual(self, result,
         """0.10866138825404954 checking_status=='b'<0'' AND foreign_worker=='b'yes''
            0.10705790652183149 checking_status=='b'<0'' AND job=='b'skilled''
            0.10371988780388462 checking_status=='b'<0'' AND other_parties=='b'none''
            0.1032107831257212 checking_status=='b'<0'' AND savings_status=='b'<100''
            0.10086921502691486 checking_status=='b'<0''
            0.10060089731478809 checking_status=='b'<0'' AND num_dependents==1.0
            0.09924929488520949 checking_status=='b'<0'' AND installment_commitment==4.0
            0.09469906064957609 checking_status=='b'<0'' AND existing_credits==1.0
            0.08675031975461583 checking_status=='b'<0'' AND own_telephone=='b'none''
            0.08528028654224418 checking_status=='b'<0'' AND other_payment_plans=='b'none''
            0.08499999999999999 checking_status=='b'<0'' AND credit_history=='b'existing paid''
            0.07816842732397161 checking_status=='b'<0'' AND residence_since==4.0
            0.07334123103829637 age: [30.0:36.0[ AND checking_status=='b'<0''
            0.07278624758728697 checking_status=='b'<0'' AND property_magnitude=='b'car''
            0.07155417527999328 checking_status=='b'<0'' AND property_magnitude=='b'no known property''
            0.06928203230275509 credit_history=='b'no credits/all paid'' AND savings_status=='b'<100''
            0.06734716791106383 credit_history=='b'no credits/all paid'' AND other_parties=='b'none''
            0.06659868328566726 checking_status=='b'<0'' AND purpose=='b'new car''
            0.06607158652139772 checking_status=='b'<0'' AND personal_status=='b'female div/dep/mar''
            0.0654846187598099 checking_status=='b'<0'' AND housing=='b'own''
            0.065 credit_history=='b'no credits/all paid''
            0.06471832459560074 checking_status=='b'<0'' AND housing=='b'rent''
            0.0646366361813647 credit_history=='b'no credits/all paid'' AND foreign_worker=='b'yes''
            0.06411591895800751 checking_status=='b'0<=X<200'' AND property_magnitude=='b'no known property''
            0.06399448505650358 other_payment_plans=='b'bank'' AND purpose=='b'new car''
            0.06394465512393448 checking_status=='b'<0'' AND personal_status=='b'male single''
            0.063727937358744 age<26.0 AND purpose=='b'new car''
            0.06365809407433175 age<26.0 AND checking_status=='b'<0''
            0.06301438731059822 age<26.0 AND savings_status=='b'<100''
            0.06272082938724895 credit_history=='b'all paid'' AND other_parties=='b'none''""")



if __name__ == "__main__":
    unittest.main()
