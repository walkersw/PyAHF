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

        A, b = self.Mesh.Affine_Map()
        print(A)
        print(b)
        
        print("in test_Affine_Map!")


if __name__ == '__main__':
    unittest.main()
