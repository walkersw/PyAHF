import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.SimplexMath import *

class TestSimplexMath(unittest.TestCase):

    # @classmethod
    # def setUpClass(self):

    # @classmethod
    # def tearDownClass(self):

    # def setUp(self):

    def tearDown(self):
        print(" ")

    def test_Affine_Map(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])

        print(vc0)
        A0, b0 = Affine_Map(vc0)
        print(A0)
        print(b0)
        J_sq = np.matmul(A0.T,A0)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 10.0, "Should be 10.0.")
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])

        print(vc1)
        A1, b1 = Affine_Map(vc1)
        print(A1)
        print(b1)
        J_sq = np.matmul(A1.T,A1)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 11.67155088, "Should be 11.67155088.")

        vc2 = np.array([vc0, vc1])
        A2, b2 = Affine_Map(vc2)
        print(A2)
        print(b2)
        A2_CHK = np.array([A0, A1])
        b2_CHK = np.array([b0, b1])
        self.assertEqual(np.array_equal(A2_CHK,A2), True, "Should be True.")
        self.assertEqual(np.array_equal(b2_CHK,b2), True, "Should be True.")

    def test_Reference_To_Cartesian(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        rc0 = np.array([0.3, 0.2])
        print(rc0)

        cart0 = Reference_To_Cartesian(vc0, rc0)
        print(cart0)
        diff0 = np.amax(np.abs(cart0 - np.array([0.6, 1.9, 1.1], dtype=CoordType)))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        rc1 = np.array([0.1, 0.7])
        print(rc1)

        cart1 = Reference_To_Cartesian(vc1, rc1)
        print(cart1)
        diff1 = np.amax(np.abs(cart1 - np.array([2.33, 0.79, 3.12], dtype=CoordType)))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        rc2 = np.array([rc0, rc1])
        cart2 = Reference_To_Cartesian(vc2, rc2)
        #print(cart2)
        cart2_CHK = np.array([cart0, cart1])
        self.assertEqual(np.array_equal(cart2_CHK,cart2), True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
