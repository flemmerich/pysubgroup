import unittest
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup import model_target
from pysubgroup.tests.t_utils import assertResultEqual


class TestGpGrowth(unittest.TestCase):
    def test_gp_growth(self):
        data = get_credit_data()
        #warnings.filterwarnings("error")
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['duration', 'credit_amount', 'class'])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['duration', 'credit_amount', 'class'])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric

        target=model_target.EMM_Likelihood(model_target.PolyRegression_ModelClass(x_name='duration', y_name='credit_amount'))
        qf = target
        task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=30, depth=2, qf=qf, constraints=[ps.MinSupportConstraint(100)])
        results = ps.GpGrowth().execute(task)

        assertResultEqual(self, results,
            """0.013797436813222048 property_magnitude=='b'car'' AND purpose=='b'radio/tv''
            0.009898734286313158 credit_history=='b'existing paid'' AND job=='b'unskilled resident''
            0.008246523912217767 employment=='b'>=7'' AND existing_credits==2.0
            0.008140633988502345 residence_since==1.0
            0.0077895968955903 installment_commitment==4.0 AND personal_status=='b'female div/dep/mar''
            0.0074121246910140584 employment=='b'4<=X<7'' AND personal_status=='b'male single''
            0.0071337090733226 employment=='b'<1'' AND existing_credits==1.0
            0.0068670509196103215 checking_status=='b'no checking'' AND residence_since==2.0
            0.006504784620278891 residence_since==2.0 AND savings_status=='b'<100''
            0.006155044522767603 own_telephone=='b'none'' AND property_magnitude=='b'real estate''
            0.005861872663834833 housing=='b'rent'' AND residence_since==4.0
            0.005860952266316298 housing=='b'own'' AND purpose=='b'furniture/equipment''
            0.005803210503193562 property_magnitude=='b'car'' AND residence_since==2.0
            0.005799115508272782 installment_commitment==4.0 AND purpose=='b'radio/tv''
            0.005784113955920163 checking_status=='b'no checking'' AND purpose=='b'radio/tv''
            0.00576020125931543 own_telephone=='b'yes'' AND personal_status=='b'female div/dep/mar''
            0.005710873446125693 age: [30.0:36.0[ AND own_telephone=='b'yes''
            0.005447977966354751 employment=='b'>=7'' AND personal_status=='b'male single''
            0.005369226054398001 credit_history=='b'critical/other existing credit'' AND own_telephone=='b'yes''
            0.0051795091578145415 checking_status=='b'<0''
            0.005142633556137799 installment_commitment==3.0 AND job=='b'skilled''
            0.005113508747833315 personal_status=='b'female div/dep/mar'' AND property_magnitude=='b'car''
            0.005076244770614837 job=='b'unskilled resident'' AND num_dependents==1.0
            0.005037829303302348 checking_status=='b'no checking'' AND employment=='b'>=7''
            0.004968598479332576 age: [26.0:30.0[ AND own_telephone=='b'none''
            0.004948240707577685 employment=='b'<1'' AND other_payment_plans=='b'none''
            0.004908520358522154 age<26.0 AND existing_credits==1.0
            0.00483781209051734 installment_commitment==3.0 AND other_payment_plans=='b'none''
            0.00476540157528357 own_telephone=='b'none'' AND purpose=='b'furniture/equipment''
            0.004700479771495154 employment=='b'1<=X<4'' AND existing_credits==2.0""")


if __name__ == '__main__':
    unittest.main()
