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

        cell_vtx = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=VtxIndType)
        cell_vtx.shape = (4,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        self.assertEqual(self.Mesh.Num_Cell(), 4, "Num_Cell should be 4.")
        self.assertEqual(self.Mesh.Top_Dim(), 2, "Top_Dim should be 2.")
        
        Cell_cap, Vtx2HF_cap = self.Mesh.Capacity()
        self.assertEqual(Cell_cap, 5, "Cell Capacity should be 5.")
        self.assertEqual(Vtx2HF_cap, 0, "Vtx2HF capacity should be 0.")

    def test_Set_Geo_Dim(self):
        self.test_Create()

        print(self.VC.coord)
        self.Mesh.Set_Geometric_Dimension(5)
        print(self.VC.coord)
        self.Mesh.Set_Geometric_Dimension(3)
        print(self.VC.coord)
        self.Mesh.Set_Geometric_Dimension(8)
        print(self.VC.coord)
        self.Mesh.Set_Geometric_Dimension(2)
        print(self.VC.coord)

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

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
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

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
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

    def test_Bary2Ref_and_back(self):
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

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        bc0 = np.array([0.1, 0.4, 0.5])
        print(bc0)
        rc0 = self.Mesh.Barycentric_To_Reference(bc0)
        print(rc0)
        self.assertEqual(np.array_equal(bc0[1:],rc0), True, "Should be True.")
        bc0_CHK = self.Mesh.Reference_To_Barycentric(rc0)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")

        bc1 = np.array([0.22, 0.48, 0.3])
        print(bc1)
        rc1 = self.Mesh.Barycentric_To_Reference(bc1)
        print(rc1)
        self.assertEqual(np.array_equal(bc1[1:],rc1), True, "Should be True.")
        bc1_CHK = self.Mesh.Reference_To_Barycentric(rc1)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")

        bc2 = np.array([bc0, bc1])
        rc2 = self.Mesh.Barycentric_To_Reference(bc2)
        #print(rc2)
        rc2_CHK = np.array([rc0, rc1])
        diff_rc2 = np.amax(np.abs(rc2 - rc2_CHK))
        self.assertEqual(diff_rc2 < 1e-15, True, "Should be True.")
        bc2_CHK = self.Mesh.Reference_To_Barycentric(rc2)
        diff_bc2 = np.amax(np.abs(bc2 - bc2_CHK))
        self.assertEqual(diff_bc2 < 1e-15, True, "Should be True.")

    def test_Bary2Cart_and_back(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(3)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0.2, 0.4,  0.5, 0.1, -0.2,  0.5, 1, 0.15,  1, -1, -0.15,  3.1, 0.5, 0.2])
        pt_coord.shape = (5,3)
        self.VC.Append(pt_coord)
        print(self.VC)

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        bc0 = np.array([0.3, 0.3, 0.4])
        print(bc0)
        cart0 = self.Mesh.Barycentric_To_Cartesian(0,bc0)
        print(cart0)
        vc0 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0,:],:]
        cart0_CHK = Barycentric_To_Cartesian(vc0, bc0)
        diff0 = np.amax(np.abs(cart0 - cart0_CHK))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        
        bc0_CHK = self.Mesh.Cartesian_To_Barycentric(0,cart0)
        #print(bc0_CHK)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")

        bc1 = np.array([0.2, 0.6, 0.2])
        print(bc1)
        cart1 = self.Mesh.Barycentric_To_Cartesian(1,bc1)
        print(cart1)
        vc1 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[1,:],:]
        cart1_CHK = Barycentric_To_Cartesian(vc1, bc1)
        diff1 = np.amax(np.abs(cart1 - cart1_CHK))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        
        bc1_CHK = self.Mesh.Cartesian_To_Barycentric(1,cart1)
        #print(bc1_CHK)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")

        bc_all = np.array([bc0, bc1, np.array([0.5, 0.2, 0.3])])
        print(bc_all)
        cart_all = self.Mesh.Barycentric_To_Cartesian(bary_coord=bc_all)
        print(cart_all)
        vc_all = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0:3,:],:]
        #print(vc_all)
        cart_all_CHK = Barycentric_To_Cartesian(vc_all, bc_all)
        diff_all = np.amax(np.abs(cart_all - cart_all_CHK))
        self.assertEqual(diff_all < 1e-15, True, "Should be True.")
        
        bc_all_CHK = self.Mesh.Cartesian_To_Barycentric(cart_coord=cart_all)
        #print(bc_all_CHK)
        diff_bc_all = np.amax(np.abs(bc_all - bc_all_CHK))
        self.assertEqual(diff_bc_all < 1e-15, True, "Should be True.")

    def test_Measures(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(3)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0.2, 0.4,  0.5, 0.1, -0.2,  0.5, 1, 0.15,  1, -1, -0.15,  3.1, 0.5, 0.2])
        pt_coord.shape = (5,3)
        self.VC.Append(pt_coord)
        print(self.VC)

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        D0 = self.Mesh.Diameter(0)
        print(D0)
        vc0 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0,:],:]
        D0_CHK = Diameter(vc0)
        err_D0 = np.amax(np.abs(D0 - D0_CHK))
        self.assertEqual(err_D0 < 1e-15, True, "Should be True.")
        V0 = self.Mesh.Volume(0)
        print(V0)
        V0_CHK = Volume(vc0)
        err_V0 = np.amax(np.abs(V0 - V0_CHK))
        self.assertEqual(err_V0 < 1e-15, True, "Should be True.")
        A0 = self.Mesh.Angles(0)
        print(A0)
        A0_CHK = Angles(vc0)
        err_A0 = np.amax(np.abs(A0 - A0_CHK))
        self.assertEqual(err_A0 < 1e-15, True, "Should be True.")
        BB_min0, BB_max0 = self.Mesh.Bounding_Box(0)
        print(BB_min0)
        print(BB_max0)
        BB_min0_CHK, BB_max0_CHK = Bounding_Box(vc0)
        err_BB_min0 = np.amax(np.abs(BB_min0 - BB_min0_CHK))
        err_BB_max0 = np.amax(np.abs(BB_max0 - BB_max0_CHK))
        self.assertEqual(err_BB_min0 < 1e-15, True, "Should be True.")
        self.assertEqual(err_BB_max0 < 1e-15, True, "Should be True.")

        D1_vec = self.Mesh.Diameter(np.array([1, 2], dtype=CellIndType))
        print(D1_vec)
        vc1 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[[1, 2],:],:]
        D1_vec_CHK = Diameter(vc1)
        err_D1 = np.amax(np.abs(D1_vec - D1_vec_CHK))
        self.assertEqual(err_D1 < 1e-15, True, "Should be True.")
        V1_vec = self.Mesh.Volume(np.array([1, 2], dtype=CellIndType))
        print(D1_vec)
        V1_vec_CHK = Volume(vc1)
        err_V1 = np.amax(np.abs(V1_vec - V1_vec_CHK))
        self.assertEqual(err_V1 < 1e-15, True, "Should be True.")
        A1_vec = self.Mesh.Angles(np.array([1, 2], dtype=CellIndType))
        print(A1_vec)
        A1_vec_CHK = Angles(vc1)
        err_A1 = np.amax(np.abs(A1_vec - A1_vec_CHK))
        self.assertEqual(err_A1 < 1e-15, True, "Should be True.")

        BB_min1_vec, BB_max1_vec = self.Mesh.Bounding_Box(np.array([1, 2], dtype=CellIndType))
        print(BB_min1_vec)
        print(BB_max1_vec)
        BB_min1_vec_CHK, BB_max1_vec_CHK = Bounding_Box(vc1)
        err_BB_min1 = np.amax(np.abs(BB_min1_vec - BB_min1_vec_CHK))
        err_BB_max1 = np.amax(np.abs(BB_max1_vec - BB_max1_vec_CHK))
        self.assertEqual(err_BB_min1 < 1e-15, True, "Should be True.")
        self.assertEqual(err_BB_max1 < 1e-15, True, "Should be True.")

    def test_Centers(self):
        del(self.Mesh)
        del(self.VC)
        self.VC   = VtxCoordType(3)
        self.Mesh = SimplexMesh(2,self.VC)
        
        self.VC.Reserve(5)
        print(" ")
        pt_coord = np.array([0, 0.2, 0.4,  0.5, 0.1, -0.2,  0.5, 1, 0.15,  1, -1, -0.15,  3.1, 0.5, 0.2])
        pt_coord.shape = (5,3)
        self.VC.Append(pt_coord)
        print(self.VC)

        cell_vtx = np.array([0, 1, 2,  1, 3, 2,  3, 4, 2], dtype=VtxIndType)
        cell_vtx.shape = (3,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        B0 = self.Mesh.Barycenter(0)
        print(B0)
        vc0 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[0,:],:]
        B0_CHK = Barycenter(vc0)
        err_B0 = np.amax(np.abs(B0 - B0_CHK))
        self.assertEqual(err_B0 < 1e-15, True, "Should be True.")
        CB0, CR0 = self.Mesh.Circumcenter(0)
        print(CB0)
        print(CR0)
        CB0_CHK, CR0_CHK = Circumcenter(vc0)
        err_CB0 = np.amax(np.abs(CB0 - CB0_CHK))
        err_CR0 = np.amax(np.abs(CR0 - CR0_CHK))
        self.assertEqual(err_CB0 < 1e-15, True, "Should be True.")
        self.assertEqual(err_CR0 < 1e-15, True, "Should be True.")
        IB0, IR0 = self.Mesh.Incenter(0)
        print(IB0)
        print(IR0)
        IB0_CHK, IR0_CHK = Incenter(vc0)
        err_IB0 = np.amax(np.abs(IB0 - IB0_CHK))
        err_IR0 = np.amax(np.abs(IR0 - IR0_CHK))
        self.assertEqual(err_IB0 < 1e-15, True, "Should be True.")
        self.assertEqual(err_IR0 < 1e-15, True, "Should be True.")
        RAT0 = self.Mesh.Shape_Regularity(0)
        print(RAT0)
        RAT0_CHK = Shape_Regularity(vc0)
        err_RAT0 = np.amax(np.abs(RAT0 - RAT0_CHK))
        self.assertEqual(err_RAT0 < 1e-15, True, "Should be True.")

        B1_vec = self.Mesh.Barycenter(np.array([1, 2], dtype=CellIndType))
        print(B1_vec)
        vc1 = self.Mesh._Vtx.coord[self.Mesh.Cell.vtx[[1, 2],:],:]
        B1_vec_CHK = Barycenter(vc1)
        err_B1 = np.amax(np.abs(B1_vec - B1_vec_CHK))
        self.assertEqual(err_B1 < 1e-15, True, "Should be True.")
        CB1_vec, CR1_vec = self.Mesh.Circumcenter(np.array([1, 2], dtype=CellIndType))
        print(CB1_vec)
        print(CR1_vec)
        CB1_vec_CHK, CR1_vec_CHK = Circumcenter(vc1)
        err_CB1 = np.amax(np.abs(CB1_vec - CB1_vec_CHK))
        err_CR1 = np.amax(np.abs(CR1_vec - CR1_vec_CHK))
        self.assertEqual(err_CB1 < 1e-15, True, "Should be True.")
        self.assertEqual(err_CR1 < 1e-15, True, "Should be True.")
        IB1_vec, IR1_vec = self.Mesh.Incenter(np.array([1, 2], dtype=CellIndType))
        print(IB1_vec)
        print(IR1_vec)
        IB1_vec_CHK, IR1_vec_CHK = Incenter(vc1)
        err_IB1 = np.amax(np.abs(IB1_vec - IB1_vec_CHK))
        err_IR1 = np.amax(np.abs(IR1_vec - IR1_vec_CHK))
        self.assertEqual(err_IB1 < 1e-15, True, "Should be True.")
        self.assertEqual(err_IR1 < 1e-15, True, "Should be True.")
        RAT1_vec = self.Mesh.Shape_Regularity(np.array([1, 2], dtype=CellIndType))
        print(RAT1_vec)
        RAT1_vec_CHK = Shape_Regularity(vc1)
        err_RAT1 = np.amax(np.abs(RAT1_vec - RAT1_vec_CHK))
        self.assertEqual(err_RAT1 < 1e-15, True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
