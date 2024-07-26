import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.MeshFactory import *

from VTKwrite.interface import unstructuredGridToVTK
from VTKwrite.vtkbin import VtkTriangle

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
        VC, Mesh = MF.Simplex_Mesh_Of_Rectangle(Pll=[0.0, 0.0], Pur=[2.0, 1.0], N0=2, N1=3)
        print(VC)
        print(Mesh)
        #VC.Print()
        #Mesh.Cell.Print()
        #Mesh.Print_Vtx2HalfFacets()

        # reference mesh data
        Point_Coord_CHK = np.array([[0.0, 0.0], [0.0, (1.0/3.0)], [0.0, (2.0/3.0)], [0.0, 1.0], [1.0, 0.0], [1.0, (1.0/3.0)], [1.0, (2.0/3.0)], [1.0, 1.0], [2.0, 0.0], [2.0, (1.0/3.0)], [2.0, (2.0/3.0)], [2.0, 1.0]], dtype=CoordType)
        Cell_CHK = np.array([[0, 4, 5], [1, 5, 6], [2, 6, 7], [4, 8, 9], [5, 9, 10], [6, 10, 11], [0, 5, 1], [1, 6, 2], [2, 7, 3], [4, 9, 5], [5, 10, 6], [6, 11, 7]], dtype=VtxIndType)
        
        Point_Coord_Diff = np.amax(np.abs(Point_Coord_CHK - VC.coord[0:VC.Size(),:]))
        self.assertEqual(Point_Coord_Diff < 1e-15, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_CHK,Mesh.Cell.vtx[0:Mesh.Num_Cell(),:]), True, "Should be True.")

        # write to a VTK file
        Px, Py, Pz, Mesh_Conn, Cell_Offsets = Mesh.Export_For_VTKwrite()
        Cell_Types = np.zeros(Cell_Offsets.size)
        Cell_Types[:] = VtkTriangle.tid
        
        # print(Px)
        # print(Px.shape)
        # print(Mesh_Conn)
        # print(Mesh_Conn.shape)
        # print(Cell_Offsets)
        # print(Cell_Offsets.shape)
        # print(Cell_Types)
        # print(Cell_Types.shape)
        
        FILE_PATH = "/mnt/c/TEMP/mesh_2D"
        unstructuredGridToVTK(FILE_PATH, Px, Py, Pz, connectivity = Mesh_Conn, offsets = Cell_Offsets, cell_types = Cell_Types)
        
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
