import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.SimplexMesh import *

class TestSimplexMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.Mesh = BaseSimplexMesh(0)

    @classmethod
    def tearDownClass(self):
        self.Mesh.Clear()

    def setUp(self):
        self.Mesh = BaseSimplexMesh(0)

    def tearDown(self):
        self.Mesh.Clear()
        print(" ")

    # def test_Size_Capacity_Dim(self):
        # del(self.Mesh)
        # self.Mesh = BaseSimplexMesh(2)
        
        # cell_vtx = np.array([0, 3, 1, 1, 2, 0, 4, 0, 2, 4, 5, 0])
        # cell_vtx.shape = (4,3)
        # self.Mesh.Append_Cell(cell_vtx)
        # print(self.Mesh)

        # self.assertEqual(self.Mesh.Num_Cell(), 4, "Num_Cell should be 4.")
        # self.assertEqual(self.Mesh.Top_Dim(), 2, "Top_Dim should be 2.")
        
        # Cell_cap, Vtx2HF_cap = self.Mesh.Capacity()
        # self.assertEqual(Cell_cap, 5, "Cell Capacity should be 5.")
        # self.assertEqual(Vtx2HF_cap, 0, "Vtx2HF capacity should be 0.")

    # def test_Reserve(self):
        # del(self.Mesh)
        # self.Mesh = BaseSimplexMesh(4)

        # num_reserve = 4
        # self.Mesh.Reserve(num_reserve)
        # Cell_cap, Vtx2HalfFacets_cap = self.Mesh.Capacity()
        # print(self.Mesh)
        # self.assertEqual(Cell_cap, num_reserve+1, "Cell capacity should be " + str(num_reserve+1) + ".")
        # self.assertEqual(Vtx2HalfFacets_cap, 0, "Vtx2HalfFacets capacity should be " + str(0) + ".")
        # _v2hfs_cap = np.rint((4+1) * num_reserve * 1.2)
        # self.assertEqual(self.Mesh._v2hfs.Capacity(), _v2hfs_cap, "_v2hfs capacity should be " + str(_v2hfs_cap) + ".")
        


if __name__ == '__main__':
    unittest.main()
