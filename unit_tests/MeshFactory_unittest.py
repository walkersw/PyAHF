import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.MeshFactory import *

from VTKwrite.interface import unstructuredGridToVTK
from VTKwrite.vtkbin import VtkTriangle, VtkTetra

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
        VC, Mesh = MF.Simplex_Mesh_Of_Rectangle(Pll=[0.0, 0.0], Pur=[2.0, 1.0], N0=2, N1=3)
        #print(VC)
        print(Mesh)
        #VC.Print()
        #Mesh.Cell.Print()
        #Mesh.Print_Vtx2HalfFacets()

        # basic checks
        consistent_sibhfs = Mesh._Check_Sibling_HalfFacets()
        self.assertEqual(consistent_sibhfs, True, "Should be True.")
        consistent_vhfs = Mesh._Check_Vtx2HalfFacets(True)
        self.assertEqual(consistent_vhfs, True, "Should be True.")

        print("Nonmanifold_Vertices:")
        Mesh.Print_Nonmanifold_Vertices()

        # reference mesh data
        Point_Coord_CHK = np.array([[0.0, 0.0], [0.0, (1.0/3.0)], [0.0, (2.0/3.0)], [0.0, 1.0], [1.0, 0.0], [1.0, (1.0/3.0)], [1.0, (2.0/3.0)], [1.0, 1.0], [2.0, 0.0], [2.0, (1.0/3.0)], [2.0, (2.0/3.0)], [2.0, 1.0]], dtype=CoordType)
        Cell_CHK = np.array([[0, 4, 5], [1, 5, 6], [2, 6, 7], [4, 8, 9], [5, 9, 10], [6, 10, 11], [0, 5, 1], [1, 6, 2], [2, 7, 3], [4, 9, 5], [5, 10, 6], [6, 11, 7]], dtype=VtxIndType)
        
        Point_Coord_Diff = np.amax(np.abs(Point_Coord_CHK - VC.coord[0:VC.Size(),:]))
        self.assertEqual(Point_Coord_Diff < 1e-15, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_CHK,Mesh.Cell.vtx[0:Mesh.Num_Cell(),:]), True, "Should be True.")

        # test the BCC option
        VC_bcc, Mesh_bcc = MF.Simplex_Mesh_Of_Rectangle(Pll=[0.0, 0.0], Pur=[2.0, 1.0], N0=2, N1=3, UseBCC=True)
        #print(VC_bcc)
        print(Mesh_bcc)
        #VC_bcc.Print()
        #Mesh_bcc.Cell.Print()
        #Mesh_bcc.Print_Vtx2HalfFacets()

        # basic checks
        consistent_sibhfs = Mesh_bcc._Check_Sibling_HalfFacets()
        self.assertEqual(consistent_sibhfs, True, "Should be True.")
        consistent_vhfs = Mesh_bcc._Check_Vtx2HalfFacets(True)
        self.assertEqual(consistent_vhfs, True, "Should be True.")

        print("Nonmanifold_Vertices:")
        Mesh_bcc.Print_Nonmanifold_Vertices()

        # reference mesh data
        Point_Coord_bcc_CHK = np.array([[0.0, 0.0], [0.0, (1.0/3.0)], [0.0, (2.0/3.0)], [0.0, 1.0], [1.0, 0.0], [1.0, (1.0/3.0)], [1.0, (2.0/3.0)], [1.0, 1.0], [2.0, 0.0], [2.0, (1.0/3.0)], [2.0, (2.0/3.0)], [2.0, 1.0], [0.5, (1.0/6.0)], [0.5, 0.5], [0.5, (5.0/6.0)], [1.5, (1.0/6.0)], [1.5, 0.5], [1.5, (5.0/6.0)]], dtype=CoordType)
        Cell_bcc_CHK = np.array([[0, 4, 12], [1, 5, 13], [2, 6, 14], [4, 8, 15], [5, 9, 16], [6, 10, 17], [4, 5, 12], [5, 6, 13], [6, 7, 14], [8, 9, 15], [9, 10, 16], [10, 11, 17], [5, 1, 12], [6, 2, 13], [7, 3, 14], [9, 5, 15], [10, 6, 16], [11, 7, 17], [1, 0, 12], [2, 1, 13], [3, 2, 14], [5, 4, 15], [6, 5, 16], [7, 6, 17]], dtype=VtxIndType)
        
        Point_Coord_bcc_Diff = np.amax(np.abs(Point_Coord_bcc_CHK - VC_bcc.coord[0:VC_bcc.Size(),:]))
        self.assertEqual(Point_Coord_bcc_Diff < 1e-15, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_bcc_CHK,Mesh_bcc.Cell.vtx[0:Mesh_bcc.Num_Cell(),:]), True, "Should be True.")


        # # write to a VTK file
        # Px, Py, Pz, Mesh_Conn, Cell_Offsets = Mesh.Export_For_VTKwrite()
        # Cell_Types = np.zeros(Cell_Offsets.size)
        # Cell_Types[:] = VtkTriangle.tid
        
        # print(Px)
        # print(Px.shape)
        # print(Mesh_Conn)
        # print(Mesh_Conn.shape)
        # print(Cell_Offsets)
        # print(Cell_Offsets.shape)
        # print(Cell_Types)
        # print(Cell_Types.shape)
        
        # FILE_PATH = "/mnt/c/TEMP/mesh_2D"
        # unstructuredGridToVTK(FILE_PATH, Px, Py, Pz, connectivity = Mesh_Conn, offsets = Cell_Offsets, cell_types = Cell_Types)


    def test_Box_Mesh(self):
        MF = MeshFactory()
        print(MF)
        #VC, Mesh = MF.Simplex_Mesh_Of_Box(Pbll=[0.0, 0.0, 0.0], Ptur=[2.0, 1.0, 0.5], N0=2, N1=1, N2=1)
        VC, Mesh = MF.Simplex_Mesh_Of_Box(Pbll=[0.0, 0.0, 0.0], Ptur=[2.0, 1.0, 0.5], N0=2, N1=3, N2=2)
        #print(VC)
        print(Mesh)
        #VC.Print()
        #Mesh.Cell.Print()
        #Mesh.Print_Vtx2HalfFacets()
        
        # basic checks
        consistent_sibhfs = Mesh._Check_Sibling_HalfFacets()
        self.assertEqual(consistent_sibhfs, True, "Should be True.")
        consistent_vhfs = Mesh._Check_Vtx2HalfFacets(True)
        self.assertEqual(consistent_vhfs, True, "Should be True.")

        print("Nonmanifold_Vertices:")
        Mesh.Print_Nonmanifold_Vertices()

        # reference mesh data
        PX_ref = 2.0*np.array([0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0, 0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0, 0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0], dtype=CoordType)
        PX_ref.shape = [PX_ref.shape[0], 1]
        PY_ref = 1.0*np.array([0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0,  0,(1/3),(2/3),1.0], dtype=CoordType)
        PY_ref.shape = [PY_ref.shape[0], 1]
        PZ_ref = 0.5*np.array([0,0,0,0,0,0,0,0,0,0,0,0,  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=CoordType)
        PZ_ref.shape = [PZ_ref.shape[0], 1]
        Point_Coord_CHK = np.hstack((PX_ref,PY_ref,PZ_ref))
        Cell_CHK = np.array([[0, 17, 1, 13], [1, 18, 2, 14], [2, 19, 3, 15], [4, 21, 5, 17], [5, 22, 6, 18], [6, 23, 7, 19], [12, 29, 13, 25], [13, 30, 14, 26], [14, 31, 15, 27], [16, 33, 17, 29], [17, 34, 18, 30], [18, 35, 19, 31], [0, 12, 17, 13], [1, 13, 18, 14], [2, 14, 19, 15], [4, 16, 21, 17], [5, 17, 22, 18], [6, 18, 23, 19], [12, 24, 29, 25], [13, 25, 30, 26], [14, 26, 31, 27], [16, 28, 33, 29], [17, 29, 34, 30], [18, 30, 35, 31], [0, 5, 1, 17], [1, 6, 2, 18], [2, 7, 3, 19], [4, 9, 5, 21], [5, 10, 6, 22], [6, 11, 7, 23], [12, 17, 13, 29], [13, 18, 14, 30], [14, 19, 15, 31], [16, 21, 17, 33], [17, 22, 18, 34], [18, 23, 19, 35], [0, 4, 5, 17], [1, 5, 6, 18], [2, 6, 7, 19], [4, 8, 9, 21], [5, 9, 10, 22], [6, 10, 11, 23], [12, 16, 17, 29], [13, 17, 18, 30], [14, 18, 19, 31], [16, 20, 21, 33], [17, 21, 22, 34], [18, 22, 23, 35], [0, 12, 16, 17], [1, 13, 17, 18], [2, 14, 18, 19], [4, 16, 20, 21], [5, 17, 21, 22], [6, 18, 22, 23], [12, 24, 28, 29], [13, 25, 29, 30], [14, 26, 30, 31], [16, 28, 32, 33], [17, 29, 33, 34], [18, 30, 34, 35], [0, 16, 4, 17], [1, 17, 5, 18], [2, 18, 6, 19], [4, 20, 8, 21], [5, 21, 9, 22], [6, 22, 10, 23], [12, 28, 16, 29], [13, 29, 17, 30], [14, 30, 18, 31], [16, 32, 20, 33], [17, 33, 21, 34], [18, 34, 22, 35]], dtype=VtxIndType)
        Point_Coord_Diff = np.amax(np.abs(Point_Coord_CHK - VC.coord[0:VC.Size(),:]))
        self.assertEqual(Point_Coord_Diff < 1e-15, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_CHK,Mesh.Cell.vtx[0:Mesh.Num_Cell(),:]), True, "Should be True.")

        # test the BCC option
        VC_bcc, Mesh_bcc = MF.Simplex_Mesh_Of_Box(Pbll=[0.0, 0.0, 0.0], Ptur=[2.0, 1.0, 0.5], N0=2, N1=3, N2=2, UseBCC=True)
        #print(VC_bcc)
        print(Mesh_bcc)
        #VC_bcc.Print()
        Mesh_bcc.Cell.Print()
        #Mesh_bcc.Print_Vtx2HalfFacets()

        # basic checks
        consistent_bcc_sibhfs = Mesh_bcc._Check_Sibling_HalfFacets()
        self.assertEqual(consistent_bcc_sibhfs, True, "Should be True.")
        consistent_bcc_vhfs = Mesh_bcc._Check_Vtx2HalfFacets(True)
        self.assertEqual(consistent_bcc_vhfs, True, "Should be True.")

        print("Nonmanifold_Vertices:")
        Mesh_bcc.Print_Nonmanifold_Vertices()

        # reference mesh data
        PX_ref = 2.0*np.array([0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0,  0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0, 0,0,0,0, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0,1.0,  0.25,0.25,0.25,0.75,0.75,0.75, 0.25,0.25,0.25,0.75,0.75,0.75,  1.25,1.25,1.25,1.25,1.25,1.25,  0.25,0.75,0.25,0.75,  0.25,0.25,0.25,0.75,0.75,0.75], dtype=CoordType)
        PX_ref.shape = [PX_ref.shape[0], 1]
        PY_ref = 1.0*np.array([(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0, 0,(1/3),(2/3),1.0,  (1/6),(1/2),(5/6), (1/6),(1/2),(5/6), (1/6),(1/2),(5/6), (1/6),(1/2),(5/6), (1/6),(1/2),(5/6), (1/6),(1/2),(5/6),  (7/6),(7/6),(7/6),(7/6),  (1/6),(1/2),(5/6), (1/6),(1/2),(5/6)], dtype=CoordType)
        PY_ref.shape = [PY_ref.shape[0], 1]
        PZ_ref = 0.5*np.array([0,0,0,0,0,0,0,0,0,0,0,  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,   1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,  0.25,0.25,0.25,0.25,0.25,0.25,  0.75,0.75,0.75,0.75,0.75,0.75,  0.25,0.25,0.25,0.75,0.75,0.75,  0.25,0.25,0.75,0.75,  1.25,1.25,1.25,1.25,1.25,1.25 ], dtype=CoordType)
        PZ_ref.shape = [PZ_ref.shape[0], 1]
        Point_Coord_bcc_CHK = np.hstack((PX_ref,PY_ref,PZ_ref))
        Cell_bcc_CHK = np.array([[35, 15, 3, 38], [36, 16, 4, 39], [37, 17, 5, 40], [38, 19, 7, 47], [39, 20, 8, 48], [40, 21, 9, 49], [41, 27, 15, 44], [42, 28, 16, 45], [43, 29, 17, 46], [44, 31, 19, 50], [45, 32, 20, 51], [46, 33, 21, 52], [35, 16, 15, 38], [36, 17, 16, 39], [37, 18, 17, 40], [38, 20, 19, 47], [39, 21, 20, 48], [40, 22, 21, 49], [41, 28, 27, 44], [42, 29, 28, 45], [43, 30, 29, 46], [44, 32, 31, 50], [45, 33, 32, 51], [46, 34, 33, 52], [35, 4, 16, 38], [36, 5, 17, 39], [37, 6, 18, 40], [38, 8, 20, 47], [39, 9, 21, 48], [40, 10, 22, 49], [41, 16, 28, 44], [42, 17, 29, 45], [43, 18, 30, 46], [44, 20, 32, 50], [45, 21, 33, 51], [46, 22, 34, 52], [35, 3, 4, 38], [36, 4, 5, 39], [37, 5, 6, 40], [38, 7, 8, 47], [39, 8, 9, 48], [40, 9, 10, 49], [41, 15, 16, 44], [42, 16, 17, 45], [43, 17, 18, 46], [44, 19, 20, 50], [45, 20, 21, 51], [46, 21, 22, 52], [35, 16, 4, 36], [36, 17, 5, 37], [37, 18, 6, 53], [38, 20, 8, 39], [39, 21, 9, 40], [40, 22, 10, 54], [41, 28, 16, 42], [42, 29, 17, 43], [43, 30, 18, 55], [44, 32, 20, 45], [45, 33, 21, 46], [46, 34, 22, 56], [35, 12, 16, 36], [36, 13, 17, 37], [37, 14, 18, 53], [38, 16, 20, 39], [39, 17, 21, 40], [40, 18, 22, 54], [41, 24, 28, 42], [42, 25, 29, 43], [43, 26, 30, 55], [44, 28, 32, 45], [45, 29, 33, 46], [46, 30, 34, 56], [35, 0, 12, 36], [36, 1, 13, 37], [37, 2, 14, 53], [38, 4, 16, 39], [39, 5, 17, 40], [40, 6, 18, 54], [41, 12, 24, 42], [42, 13, 25, 43], [43, 14, 26, 55], [44, 16, 28, 45], [45, 17, 29, 46], [46, 18, 30, 56], [35, 4, 0, 36], [36, 5, 1, 37], [37, 6, 2, 53], [38, 8, 4, 39], [39, 9, 5, 40], [40, 10, 6, 54], [41, 16, 12, 42], [42, 17, 13, 43], [43, 18, 14, 55], [44, 20, 16, 45], [45, 21, 17, 46], [46, 22, 18, 56], [35, 11, 15, 41], [36, 12, 16, 42], [37, 13, 17, 43], [38, 15, 19, 44], [39, 16, 20, 45], [40, 17, 21, 46], [41, 23, 27, 57], [42, 24, 28, 58], [43, 25, 29, 59], [44, 27, 31, 60], [45, 28, 32, 61], [46, 29, 33, 62], [35, 12, 11, 41], [36, 13, 12, 42], [37, 14, 13, 43], [38, 16, 15, 44], [39, 17, 16, 45], [40, 18, 17, 46], [41, 24, 23, 57], [42, 25, 24, 58], [43, 26, 25, 59], [44, 28, 27, 60], [45, 29, 28, 61], [46, 30, 29, 62], [35, 16, 12, 41], [36, 17, 13, 42], [37, 18, 14, 43], [38, 20, 16, 44], [39, 21, 17, 45], [40, 22, 18, 46], [41, 28, 24, 57], [42, 29, 25, 58], [43, 30, 26, 59], [44, 32, 28, 60], [45, 33, 29, 61], [46, 34, 30, 62], [35, 15, 16, 41], [36, 16, 17, 42], [37, 17, 18, 43], [38, 19, 20, 44], [39, 20, 21, 45], [40, 21, 22, 46], [41, 27, 28, 57], [42, 28, 29, 58], [43, 29, 30, 59], [44, 31, 32, 60], [45, 32, 33, 61], [46, 33, 34, 62]], dtype=VtxIndType)
        Point_Coord_bcc_Diff = np.amax(np.abs(Point_Coord_bcc_CHK - VC_bcc.coord[0:VC_bcc.Size(),:]))
        self.assertEqual(Point_Coord_bcc_Diff < 1e-15, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_bcc_CHK,Mesh_bcc.Cell.vtx[0:Mesh_bcc.Num_Cell(),:]), True, "Should be True.")

        # write to a VTK file
        Px, Py, Pz, Mesh_Conn, Cell_Offsets = Mesh_bcc.Export_For_VTKwrite()
        Cell_Types = np.zeros(Cell_Offsets.size)
        Cell_Types[:] = VtkTetra.tid
        
        print(Px)
        print(Px.shape)
        print(Mesh_Conn)
        print(Mesh_Conn.shape)
        print(Cell_Offsets)
        print(Cell_Offsets.shape)
        print(Cell_Types)
        print(Cell_Types.shape)
        
        FILE_PATH = "/mnt/c/TEMP/mesh_3D"
        unstructuredGridToVTK(FILE_PATH, Px, Py, Pz, connectivity = Mesh_Conn, offsets = Cell_Offsets, cell_types = Cell_Types)












if __name__ == '__main__':
    unittest.main()
