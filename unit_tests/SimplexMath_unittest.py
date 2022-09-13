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

    def test_Ref2Cart_and_back(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        rc0 = np.array([0.3, 0.2])
        print(rc0)

        cart0 = Reference_To_Cartesian(vc0, rc0)
        print(cart0)
        diff0 = np.amax(np.abs(cart0 - np.array([0.6, 1.9, 1.1], dtype=CoordType)))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        rc0_CHK = Cartesian_To_Reference(vc0, cart0)
        diff_rc0 = np.amax(np.abs(rc0 - rc0_CHK))
        self.assertEqual(diff_rc0 < 1e-15, True, "Should be True.")
        #print(rc0_CHK)
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        rc1 = np.array([0.1, 0.7])
        print(rc1)

        cart1 = Reference_To_Cartesian(vc1, rc1)
        print(cart1)
        diff1 = np.amax(np.abs(cart1 - np.array([2.33, 0.79, 3.12], dtype=CoordType)))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        rc1_CHK = Cartesian_To_Reference(vc1, cart1)
        diff_rc1 = np.amax(np.abs(rc1 - rc1_CHK))
        self.assertEqual(diff_rc1 < 1e-15, True, "Should be True.")
        #print(rc1_CHK)

        vc2 = np.array([vc0, vc1])
        rc2 = np.array([rc0, rc1])
        cart2 = Reference_To_Cartesian(vc2, rc2)
        #print(cart2)
        cart2_CHK = np.array([cart0, cart1])
        self.assertEqual(np.array_equal(cart2_CHK,cart2), True, "Should be True.")

        rc2_CHK = Cartesian_To_Reference(vc2, cart2)
        diff_rc2 = np.amax(np.abs(rc2 - rc2_CHK))
        self.assertEqual(diff_rc2 < 1e-15, True, "Should be True.")

    def test_Bary2Ref_and_back(self):
        bc0 = np.array([0.5, 0.3, 0.2])
        print(bc0)
        rc0 = Barycentric_To_Reference(bc0)
        print(rc0)
        self.assertEqual(np.array_equal(bc0[1:],rc0), True, "Should be True.")
        bc0_CHK = Reference_To_Barycentric(rc0)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")

        bc1 = np.array([0.2, 0.1, 0.7])
        print(bc1)
        rc1 = Barycentric_To_Reference(bc1)
        print(rc1)
        self.assertEqual(np.array_equal(bc1[1:],rc1), True, "Should be True.")
        bc1_CHK = Reference_To_Barycentric(rc1)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")

        bc2 = np.array([bc0, bc1])
        rc2 = Barycentric_To_Reference(bc2)
        #print(rc2)
        rc2_CHK = np.array([rc0, rc1])
        diff_rc2 = np.amax(np.abs(rc2 - rc2_CHK))
        self.assertEqual(diff_rc2 < 1e-15, True, "Should be True.")
        bc2_CHK = Reference_To_Barycentric(rc2)
        diff_bc2 = np.amax(np.abs(bc2 - bc2_CHK))
        self.assertEqual(diff_bc2 < 1e-15, True, "Should be True.")

    def test_Bary2Cart_and_back(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        bc0 = np.array([0.5, 0.3, 0.2])
        print(bc0)

        cart0 = Barycentric_To_Cartesian(vc0, bc0)
        print(cart0)
        diff0 = np.amax(np.abs(cart0 - np.array([0.6, 1.9, 1.1], dtype=CoordType)))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        bc0_CHK = Cartesian_To_Barycentric(vc0, cart0)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")
        #print(bc0_CHK)
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        bc1 = np.array([0.2, 0.1, 0.7])
        print(bc1)

        cart1 = Barycentric_To_Cartesian(vc1, bc1)
        print(cart1)
        diff1 = np.amax(np.abs(cart1 - np.array([2.33, 0.79, 3.12], dtype=CoordType)))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        bc1_CHK = Cartesian_To_Barycentric(vc1, cart1)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")
        #print(bc1_CHK)

        vc2 = np.array([vc0, vc1])
        bc2 = np.array([bc0, bc1])
        cart2 = Barycentric_To_Cartesian(vc2, bc2)
        #print(cart2)
        cart2_CHK = np.array([cart0, cart1])
        self.assertEqual(np.array_equal(cart2_CHK,cart2), True, "Should be True.")

        bc2_CHK = Cartesian_To_Barycentric(vc2, cart2)
        diff_bc2 = np.amax(np.abs(bc2 - bc2_CHK))
        self.assertEqual(diff_bc2 < 1e-15, True, "Should be True.")

    def test_Measures(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        D0 = Diameter(vc0)
        self.assertEqual(np.abs(D0 - 5.0) < 1e-15, True, "Should be True.")
        BB_min0, BB_max0 = Bounding_Box(vc0)
        self.assertEqual(np.array_equal(BB_min0,np.array([0.0, 0.0, 1.1])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max0,np.array([2.0, 5.0, 1.1])), True, "Should be True.")
        V0 = Volume(vc0)
        #print(V0)
        self.assertEqual(np.abs(V0 - 5.0) < 1e-15, True, "Should be True.")
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        D1 = Diameter(vc1)
        self.assertEqual(np.abs(D1 - 5.707889277132134) < 1e-15, True, "Should be True.")
        BB_min1, BB_max1 = Bounding_Box(vc1)
        self.assertEqual(np.array_equal(BB_min1,np.array([0.2, 0.3, -0.4])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max1,np.array([3.1, 2.4, 4.5])), True, "Should be True.")
        V1 = Volume(vc1)
        #print(V1)
        self.assertEqual(np.abs(V1 - 5.835775441190314) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        D2 = Diameter(vc2)
        D2_CHK = np.array([D0, D1])
        diff_D2 = np.amax(np.abs(D2 - D2_CHK))
        self.assertEqual(diff_D2 < 1e-15, True, "Should be True.")
        BB_min2, BB_max2 = Bounding_Box(vc2)
        self.assertEqual(np.array_equal(BB_min2,np.array([BB_min0, BB_min1])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max2,np.array([BB_max0, BB_max1])), True, "Should be True.")
        V2 = Volume(vc2)
        V2_CHK = np.array([V0, V1])
        diff_V2 = np.amax(np.abs(V2 - V2_CHK))
        self.assertEqual(diff_V2 < 1e-15, True, "Should be True.")




        print("HERE!")


if __name__ == '__main__':
    unittest.main()
