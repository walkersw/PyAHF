import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.BasicClasses import *

class TestBasicClasses(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.Cell = CellSimplexType(0)
        self.VC   = VtxCoordType(1)

    @classmethod
    def tearDownClass(self):
        self.Cell.Clear()
        self.VC.Clear()

    def setUp(self):
        self.Cell = CellSimplexType(0)
        self.VC   = VtxCoordType(1)

    def tearDown(self):
        self.Cell.Clear()
        self.VC.Clear()

    def test_Size_and_Dim(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(3)
        self.VC   = VtxCoordType(5)

        self.Cell.Reserve(10)
        self.VC.Reserve(20)
        self.assertEqual(self.Cell.Size(), 0, "Size should be 0.")
        self.assertEqual(self.VC.Size(), 0, "Size should be 0.")
        
        self.assertEqual(self.Cell.Dim(), 3, "Dim should be 3.")
        self.assertEqual(self.VC.Dim(), 5, "Dim should be 5.")

    def test_Reserve(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(4)
        self.VC   = VtxCoordType(6)

        num_reserve = 4
        self.Cell.Reserve(num_reserve)
        self.VC.Reserve(num_reserve)
        self.assertEqual(self.Cell.vtx.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")
        self.assertEqual(self.Cell.halffacet.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")
        self.assertEqual(self.VC.coord.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")

    def test_Append_and_Set(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(3)
        self.VC   = VtxCoordType(3)

        self.Cell.Reserve(5)
        print(" ")
        
        self.Cell.Append([1, 2, 3, 4])
        self.Cell.Append([4, 23, 88])
        print(self.Cell)

        new_cell_vtx = [2, 5, 4, 11, 6, 88, 9, 13, 1, 4, 90, 74, 23, 45, 71, 55]
        self.Cell.Append_Batch(4, new_cell_vtx)

        print(self.Cell.vtx)
        print(self.Cell)

        self.Cell.Set(3, [57, 14, 33, 48])
        print(self.Cell.vtx)

        self.assertEqual(np.array_equal(self.Cell.vtx[2],[6, 88, 9, 13]), True, "Should be [6, 88, 9, 13].")
        self.assertEqual(self.Cell.Size(), 5, "Should be 5.")
        
        uv = self.Cell.Get_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,[1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88]), True, "Should be [1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88].")
        
        self.VC.Reserve(5)
        print(" ")
        
        self.VC.Append([0.2, 1.4, -9.4])
        self.VC.Append([4.4, 2.3])
        print(self.VC)

        new_vtx_coord = [3.1, -2.5, 7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, 4.7, 10.6, -5.1]
        self.VC.Append_Batch(4, new_vtx_coord)

        print(self.VC.coord)
        print(self.VC)

        self.VC.Set(3, [8.8, -6.7, 2.7])
        print(self.VC.coord)

        self.assertEqual(np.array_equal(self.VC.coord[2],[-1.2, 4.4, 6.6]), True, "Should be [-1.2, 4.4, 6.6].")
        self.assertEqual(np.array_equal(self.VC.coord[4],[4.7, 10.6, -5.1]), True, "Should be [4.7, 10.6, -5.1].")
        self.assertEqual(self.VC.Size(), 5, "Should be 5.")

    def test_Print(self):
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)

        cell_data = [1, 6, 2, 18, 13, 9, 34, 12, 43, 7, 12, 16, 99, 39, 19]
        self.Cell.Set_All(5, cell_data)
        print(" ")
        self.Cell.Print()
        
        unique_vertices = self.Cell.Get_Unique_Vertices()
        self.assertEqual( np.array_equal(unique_vertices,[1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99]), True, "Should be [1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99].")
        self.Cell.Print_Unique_Vertices()

        coord_data = [7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, -19.4, 12.6, 35.7, -66.7, 32.8]
        self.VC.Set_All(6, coord_data)
        self.assertEqual(np.array_equal(self.VC.coord[3],[10.1, -19.4]), True, "Should be [10.1, -19.4].")
        print(" ")
        self.VC.Print()

    def test_MeshEdgeType(self):
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)
        print(" ")

        Edge_array = np.full(6, NULL_MeshEdge, dtype=MeshEdgeType)
        Edge_array[0] = (12, 14)
        Edge_array[1] = (1, 5)
        Edge_array[2] = (4, 23)
        Edge_array[3] = (3, 11)
        Edge_array[4] = (34, 16)
        Edge_array[5] = (45, 28)

        print(Edge_array)
        
        EE2 = np.array((4, 23), dtype=MeshEdgeType)
        self.assertEqual(Edge_array[2]==EE2, True, "Should be (4, 23).")
        self.assertEqual(Edge_array[4]['v1']==16, True, "Should be 16.")

if __name__ == '__main__':
    unittest.main()
