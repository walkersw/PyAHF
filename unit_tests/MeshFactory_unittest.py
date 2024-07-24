import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.MeshFactory import *

class TestMeshFactory(unittest.TestCase):

    # @classmethod
    # def setUpClass(self):

    # @classmethod
    # def tearDownClass(self):

    # def setUp(self):

    def tearDown(self):
        print(" ")

    def test_Rectangle_Mesh(self):
        MF = MeshFactory()
        print(MF)
        #Simplex_Mesh_Of_Rectangle(self, Pll=[0.0, 0.0], Pur=[1.0, 1.0], N0=10, N1=10, UseBCC=False):
        VC, Mesh = MF.Simplex_Mesh_Of_Rectangle()
        #print(VC)
        print(Mesh)
        #Mesh.Cell.Print()
        #Mesh.Print_Vtx2HalfFacets()

        # vc2 = np.array([vc0, vc1])
        # A2, b2 = Affine_Map(vc2)
        # print(A2)
        # print(b2)
        # A2_CHK = np.array([A0, A1])
        # b2_CHK = np.array([b0, b1])
        # self.assertEqual(np.array_equal(A2_CHK,A2), True, "Should be True.")
        # self.assertEqual(np.array_equal(b2_CHK,b2), True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
