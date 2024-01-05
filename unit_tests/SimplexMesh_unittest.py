import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.SimplexMesh import *
from ahf.SimplexMath import *

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

    def test_Ref2Cart_and_back(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(3)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0, 0.1,  0.5, 0, -0.1,  0.5, 1, 0.05,  1, -1, -0.05,  3, 0.5, 0.15])
        pt_coord.shape = (5,3)
        self.VC.Append(pt_coord)
        print(self.VC)

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=CellIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        rc0 = np.array([0.3, 0.2])
        print(rc0)
        cart0 = self.Mesh.Reference_To_Cartesian(0,rc0)
        print(cart0)
        vc0 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0,:],:]
        cart0_CHK = Reference_To_Cartesian(vc0, rc0)
        diff0 = np.amax(np.abs(cart0 - cart0_CHK))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        
        rc0_CHK = self.Mesh.Cartesian_To_Reference(0,cart0)
        #print(rc0_CHK)
        diff_rc0 = np.amax(np.abs(rc0 - rc0_CHK))
        self.assertEqual(diff_rc0 < 1e-15, True, "Should be True.")

        rc1 = np.array([0.1, 0.7])
        print(rc1)
        cart1 = self.Mesh.Reference_To_Cartesian(1,rc1)
        print(cart1)
        vc1 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[1,:],:]
        cart1_CHK = Reference_To_Cartesian(vc1, rc1)
        diff1 = np.amax(np.abs(cart1 - cart1_CHK))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        
        rc1_CHK = self.Mesh.Cartesian_To_Reference(1,cart1)
        #print(rc1_CHK)
        diff_rc1 = np.amax(np.abs(rc1 - rc1_CHK))
        self.assertEqual(diff_rc1 < 1e-15, True, "Should be True.")

        rc_all = np.array([rc0, rc1, np.array([0.5, 0.2])])
        print(rc_all)
        cart_all = self.Mesh.Reference_To_Cartesian(ref_coord=rc_all)
        print(cart_all)
        vc_all = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0:3,:],:]
        #print(vc_all)
        cart_all_CHK = Reference_To_Cartesian(vc_all, rc_all)
        diff_all = np.amax(np.abs(cart_all - cart_all_CHK))
        self.assertEqual(diff_all < 1e-15, True, "Should be True.")
        
        rc_all_CHK = self.Mesh.Cartesian_To_Reference(cart_coord=cart_all)
        #print(rc_all_CHK)
        diff_rc_all = np.amax(np.abs(rc_all - rc_all_CHK))
        self.assertEqual(diff_rc_all < 1e-15, True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
