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
        vtx_coord = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])

        print(vtx_coord)
        A, b = Affine_Map(vtx_coord)
        print(A)
        print(b)
        J_sq = np.matmul(A.T,A)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 10.0, "Should be 10.0.")
        
        vtx_coord = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])

        print(vtx_coord)
        A, b = Affine_Map(vtx_coord)
        print(A)
        print(b)
        J_sq = np.matmul(A.T,A)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 11.67155088, "Should be 11.67155088.")





if __name__ == '__main__':
    unittest.main()
