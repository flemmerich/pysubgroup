import tempfile
import os
import unittest

import pickle
import pysubgroup as ps


class TestRelationsMethods(unittest.TestCase):

    def test_pickle(self):

        A1 = ps.EqualitySelector("A", 1)
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'test.pickle')
            with open(f_name, 'wb') as f:
                pickle.dump(A1, f)

            with open(f_name, 'rb') as f:
                A2 = pickle.load(f)

            assert A1 == A2


if __name__ == '__main__':
    unittest.main()
