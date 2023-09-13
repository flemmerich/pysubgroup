import unittest



import pysubgroup as ps



class TestGeneralisationAwareQf(unittest.TestCase):
    def test_is_satisfied_dict(self):
        constr = ps.MinSupportConstraint(10)
        self.assertTrue(constr.is_satisfied(None, {"size_sg": 12}, None))
        self.assertFalse(constr.is_satisfied(None, {"size_sg": 9}, None))


if __name__ == "__main__":
    unittest.main()
