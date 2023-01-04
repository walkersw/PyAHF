import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.SimplexMesh import *

class TestSimplexMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.VC   = VtxCoordType(1)
        self.Mesh = SimplexMesh(0,self.VC)

    @classmethod
    def tearDownClass(self):
        self.Mesh.Clear()

    def setUp(self):
        self.VC   = VtxCoordType(1)
        self.Mesh = SimplexMesh(0,self.VC)

    def tearDown(self):
        self.Mesh.Clear()
        print(" ")

    def test_Create(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(2)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5])
        pt_coord.shape = (5,2)
        self.VC.Append(pt_coord)
        print(self.VC)
        self.assertEqual(self.VC.Size(), 5, "VC.Size() should be 5.")

        cell_vtx = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=CellIndType)
        cell_vtx.shape = (4,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        self.assertEqual(self.Mesh.Num_Cell(), 4, "Num_Cell should be 4.")
        self.assertEqual(self.Mesh.Top_Dim(), 2, "Top_Dim should be 2.")
        
        Cell_cap, Vtx2HF_cap = self.Mesh.Capacity()
        self.assertEqual(Cell_cap, 5, "Cell Capacity should be 5.")
        self.assertEqual(Vtx2HF_cap, 0, "Vtx2HF capacity should be 0.")

    def test_Affine_Map(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(2)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0,  0.5, 0,  0.5, 1,  1, -1,  3, 0.5])
        pt_coord.shape = (5,2)
        self.VC.Append(pt_coord)
        print(self.VC)

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=CellIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        A0, b0 = self.Mesh.Affine_Map(0)
        #print(A0)
        #print(b0)
        A0_CHK = np.array([[0.5, 0.5], [0, 1]], dtype=RealType)
        b0_CHK = np.array([[0.0], [0.0]], dtype=RealType)
        self.assertEqual(np.array_equal(A0_CHK,A0), True, "Should be True.")
        self.assertEqual(np.array_equal(b0_CHK,b0), True, "Should be True.")
        
        A1, b1 = self.Mesh.Affine_Map(1)
        #print(A1)
        #print(b1)
        A1_CHK = np.array([[0.5, 0], [-1, 1]], dtype=RealType)
        b1_CHK = np.array([[0.5], [0.0]], dtype=RealType)
        self.assertEqual(np.array_equal(A1_CHK,A1), True, "Should be True.")
        self.assertEqual(np.array_equal(b1_CHK,b1), True, "Should be True.")

        A2, b2 = self.Mesh.Affine_Map(2)
        #print(A2)
        #print(b2)
        A2_CHK = np.array([[2, -0.5], [1.5, 2]], dtype=RealType)
        b2_CHK = np.array([[1], [-1]], dtype=RealType)
        self.assertEqual(np.array_equal(A2_CHK,A2), True, "Should be True.")
        self.assertEqual(np.array_equal(b2_CHK,b2), True, "Should be True.")

        A_all, b_all = self.Mesh.Affine_Map()
        A_all_CHK = np.array([A0_CHK, A1_CHK, A2_CHK])
        b_all_CHK = np.array([b0_CHK, b1_CHK, b2_CHK])
        self.assertEqual(np.array_equal(A_all_CHK,A_all), True, "Should be True.")
        self.assertEqual(np.array_equal(b_all_CHK,b_all), True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
