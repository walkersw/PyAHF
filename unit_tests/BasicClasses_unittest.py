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
        print(" ")

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
        self.assertEqual(self.Cell.vtx.shape[0], num_reserve+1, "Reserved size should be " + str(num_reserve+1) + ".")
        self.assertEqual(self.Cell.halffacet.shape[0], num_reserve+1, "Reserved size should be " + str(num_reserve+1) + ".")
        self.assertEqual(self.VC.coord.shape[0], num_reserve+1, "Reserved size should be " + str(num_reserve+1) + ".")
        self.assertEqual(self.Cell.Capacity(), num_reserve+1, "Capacity should be " + str(num_reserve+1) + ".")
        self.assertEqual(self.VC.Capacity(), num_reserve+1, "Reserved size should be " + str(num_reserve+1) + ".")

    def test_Append_and_Set(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(3)
        self.VC   = VtxCoordType(3)

        self.Cell.Reserve(5)
        print(" ")
        
        self.Cell.Append(np.array([1, 2, 3, 4]))
        self.Cell.Append(np.array([4, 23, 88]))
        self.Cell.Print()

        new_cell_vtx = np.array([2, 5, 4, 11, 6, 88, 9, 13, 1, 4, 90, 74, 23, 45, 71, 55])
        new_cell_vtx.shape = (4,4)
        self.Cell.Append(new_cell_vtx)

        self.Cell.Print()
        #print(self.Cell.vtx)
        #print(self.Cell)

        self.Cell.Set(3, np.array([57, 14, 33, 48]))
        self.Cell.Print()

        self.assertEqual(np.array_equal(self.Cell.vtx[2],[6, 88, 9, 13]), True, "Should be [6, 88, 9, 13].")
        self.assertEqual(self.Cell.Size(), 5, "Should be 5.")
        
        uv = self.Cell.Get_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,[1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88]), True, "Should be [1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88].")
        
        Num_Vtx = self.Cell.Num_Vtx()
        self.assertEqual(Num_Vtx, 18, "Should be 18.")
        Max_vi = self.Cell.Max_Vtx_Index()
        self.assertEqual(Max_vi, 88, "Should be 88.")
        
        self.VC.Reserve(5)
        print(" ")
        
        self.VC.Append(np.array([0.2, 1.4, -9.4]))
        self.VC.Append(np.array([4.4, 2.3]))
        print(self.VC)

        new_vtx_coord = np.array([3.1, -2.5, 7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, 4.7, 10.6, -5.1])
        new_vtx_coord.shape = (4,3)
        self.VC.Append(new_vtx_coord)

        print(self.VC.coord)
        print(self.VC)

        self.VC.Set(3, np.array([8.8, -6.7, 2.7]))
        print(self.VC.coord)

        self.assertEqual(np.array_equal(self.VC.coord[2],[-1.2, 4.4, 6.6]), True, "Should be [-1.2, 4.4, 6.6].")
        self.assertEqual(np.array_equal(self.VC.coord[4],[4.7, 10.6, -5.1]), True, "Should be [4.7, 10.6, -5.1].")
        self.assertEqual(self.VC.Size(), 5, "Should be 5.")

    def test_Change_Vertex_Geo_Dim(self):
        del(self.Cell)
        del(self.VC)
        self.VC = VtxCoordType(3)

        vc_rand = np.random.rand(5,3)
        self.VC.Set(vc_rand)
        print(self.VC.coord)
        orig_VC_coord = np.copy(self.VC.coord)
        
        self.VC.Change_Dimension(5)
        print(self.VC.coord)
        self.assertEqual(np.array_equal(self.VC.coord[:,0:3],orig_VC_coord), True, "Should be equal.")
        self.assertEqual(np.linalg.norm(self.VC.coord[:,3:]), 0, "Should be 0.")

        self.VC.Change_Dimension(2)
        print(self.VC.coord)
        self.assertEqual(np.array_equal(self.VC.coord,orig_VC_coord[:,0:2]), True, "Should be equal.")

    def test_Reindex(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(3)
        self.VC   = VtxCoordType(3)

        new_cell_vtx = np.array([0, 3, 5, 1, 4, 12, 9, 6, 88, 62, 54, 72, 99, 120, 101, 154])
        new_cell_vtx.shape = (4,4)
        self.Cell.Append(new_cell_vtx)
        self.Cell.Print()
        
        new_indices = np.zeros(155)
        lin_indices = np.linspace(0, 15, num=16, dtype=VtxIndType)
        new_cell_vtx.shape = (16,)
        new_indices[new_cell_vtx] = lin_indices
        # print(new_indices)
        # print(lin_indices)
        self.Cell.Reindex_Vertices(new_indices)
        self.Cell.Print()
        CHK_Cell_vtx = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        self.assertEqual(np.array_equal(self.Cell.vtx[0:4][:],CHK_Cell_vtx), \
                         True, "Should be [[ 0  1  2  3], [ 4  5  6  7], [ 8  9 10 11], [12 13 14 15]].")

        vtx_coord = np.array([3.1, -2.5, 7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, 4.7, 10.6, -5.1])
        vtx_coord.shape = (4,3)
        self.VC.Set(vtx_coord)
        self.VC.Print()

        old_indices = np.array([3, 1, 2, 0], dtype=VtxIndType)
        new_indices = np.array([2, 4, 6, 1], dtype=VtxIndType)
        self.VC.Reindex_Vertices(old_indices, new_indices)
        self.VC.Print()
        self.assertEqual(np.array_equal(self.VC.coord[1],[3.1, -2.5, 7.3]), True, "Should be [3.1, -2.5, 7.3].")
        self.assertEqual(np.array_equal(self.VC.coord[6],[3.6, -7.8, 10.1]), True, "Should be [3.6, -7.8, 10.1].")

    def test_Bounding_Box(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)

        vtx_coord = np.array([3.1, -2.5, -7.3, 1.2, 4.4, 6.6, -3.6, 7.8, 10.1, -4.7, -6.6, -5.1])
        vtx_coord.shape = (6,2)
        self.VC.Set(vtx_coord)
        self.VC.Print()

        BB_min, BB_max = self.VC.Bounding_Box()
        self.assertEqual(np.array_equal(BB_min,[-7.3, -5.1]), True, "Should be [-7.3, -5.1].")
        self.assertEqual(np.array_equal(BB_max,[10.1,  7.8]), True, "Should be [10.1,  7.8].")
        VI = np.array([3, 0, 1])
        BB_min, BB_max = self.VC.Bounding_Box(VI)
        self.assertEqual(np.array_equal(BB_min,[-7.3, -2.5]), True, "Should be [-7.3, -2.5].")
        self.assertEqual(np.array_equal(BB_max,[ 3.1,  7.8]), True, "Should be [ 3.1,  7.8].")

    def test_Baby_Methods(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(3)
        self.VC   = VtxCoordType(3)
        
        Cell_0D = CellSimplexType(0)
        Cell_0D.Append(np.array([88, 4, 21, 14]))
        print(Cell_0D)
        self.assertEqual(Cell_0D.Adj_Vertices_In_Facet_Equal(np.array([]), np.array([])), True, "Should be True.")
        self.assertEqual(Cell_0D.Get_Vertex_With_Largest_Index_In_Facet(np.array([88]), 0), 88, "Should be 88.")
        self.assertEqual(np.array_equal(Cell_0D.Get_Adj_Vertices_In_Facet(np.array([]), 34),np.array([])), True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_0D.Get_Global_Vertices_In_Facet(np.array([88]), 0),np.array([])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_0D.Get_Local_Facets_Sharing_Local_Vertex(0),np.array([])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_0D.Get_Local_Vertices_Of_Local_Facet(0),np.array([])), \
                         True, "Should be True.")
        self.assertEqual(Cell_0D.Get_Local_Vertex_Index_In_Cell(188, [32])==NULL_Small, True, "Should be True.")
        self.assertEqual(Cell_0D.Get_Local_Vertex_Index_In_Cell(32, [32])==0, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_0D.Vtx2Adjacent(5, 2, 0),np.array([])), True, "Should be True.")

        Cell_1D = CellSimplexType(1)
        cv = np.array([0, 3, 4, 12, 88, 62, 99, 120])
        cv.shape = (4,2)
        Cell_1D.Append(cv)
        print(Cell_1D)
        self.assertEqual(Cell_1D.Adj_Vertices_In_Facet_Equal(np.array([]), np.array([])), True, "Should be True.")
        self.assertEqual(Cell_1D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35]), 1)==78, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_1D.Get_Adj_Vertices_In_Facet(np.array([67]), 34),np.array([])), True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_1D.Get_Global_Vertices_In_Facet(np.array([23, 34]), 0),np.array([34])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_1D.Get_Local_Facets_Sharing_Local_Vertex(1),np.array([0])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_1D.Get_Local_Vertices_Of_Local_Facet(0),np.array([1])), \
                         True, "Should be True.")
        self.assertEqual(Cell_1D.Get_Local_Vertex_Index_In_Cell(188, [19, 56])==NULL_Small, True, "Should be True.")
        self.assertEqual(Cell_1D.Get_Local_Vertex_Index_In_Cell(56, [19, 56])==1, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_1D.Vtx2Adjacent(87, 2, 1),np.array([])), True, "Should be True.")
        
        Cell_2D = CellSimplexType(2)
        cv = np.array([0, 3, 5, 4, 12, 9, 88, 62, 54, 99, 120, 101])
        cv.shape = (4,3)
        Cell_2D.Append(cv)
        print(Cell_2D)
        self.assertEqual(Cell_2D.Adj_Vertices_In_Facet_Equal(np.array([12]), np.array([13])), False, "Should be False.")
        self.assertEqual(Cell_2D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 66]), 0)==66, True, "Should be True.")
        self.assertEqual(Cell_2D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 66]), 1)==78, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Get_Adj_Vertices_In_Facet(np.array([18, 91]), 34),np.array([NULL_Vtx])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Get_Adj_Vertices_In_Facet(np.array([18, 91]), 18),np.array([91])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Get_Global_Vertices_In_Facet(np.array([53, 34, 12]), 1),np.array([53, 12])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Get_Local_Facets_Sharing_Local_Vertex(2),np.array([0, 1])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Get_Local_Vertices_Of_Local_Facet(1),np.array([0, 2])), \
                         True, "Should be True.")
        self.assertEqual(Cell_2D.Get_Local_Vertex_Index_In_Cell(188, [91, 29, 56])==NULL_Small, True, "Should be True.")
        self.assertEqual(Cell_2D.Get_Local_Vertex_Index_In_Cell(29, [91, 29, 56])==1, True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(87, 3, 1),np.array([NULL_Vtx])), True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(101, 3, 1),np.array([99])), True, "Should be True.")
        self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(99, 3, 2),np.array([120])), True, "Should be True.")
        
        self.Cell.Reserve(5)
        print(" ")
        
        cv = np.array([0, 3, 5, 1, 4, 12, 9, 6, 88, 62, 54, 72, 99, 120, 101, 154])
        cv.shape = (4,4)
        self.Cell.Append(cv)
        print(self.Cell)
        self.assertEqual(self.Cell.Adj_Vertices_In_Facet_Equal(np.array([12, 34]), np.array([13, 34])), False, "Should be False.")
        self.assertEqual(self.Cell.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 17, 66]), 2)==78, True, "Should be True.")
        self.assertEqual(self.Cell.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 17, 66]), 0)==66, True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Get_Adj_Vertices_In_Facet(np.array([18, 91, 23]), 34), \
                         np.array([NULL_Vtx, NULL_Vtx])), True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Get_Adj_Vertices_In_Facet(np.array([18, 91, 23]), 18), \
                         np.array([91, 23])), True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Get_Global_Vertices_In_Facet(np.array([53, 5, 12, 8]), 2),np.array([53, 5, 8])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Get_Local_Facets_Sharing_Local_Vertex(3),np.array([0, 1, 2])), \
                         True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Get_Local_Vertices_Of_Local_Facet(1),np.array([0, 2, 3])), \
                         True, "Should be True.")
        self.assertEqual(self.Cell.Get_Local_Vertex_Index_In_Cell(188, [91, 29, 13, 56])==NULL_Small, True, "Should be True.")
        self.assertEqual(self.Cell.Get_Local_Vertex_Index_In_Cell(13, [91, 29, 13, 56])==2, True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(87, 1, 2),np.array([NULL_Vtx, NULL_Vtx])), True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(9, 1, 1),np.array([4, 6])), True, "Should be True.")
        self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(4, 1, 2),np.array([12, 6])), True, "Should be True.")

    def test_Attached_and_Connected_and_FreeBdy(self):
        """Example Mesh:

            V1 +-------------------+ V3
               |\      (0,0)       |
               |  \           T0   |
               |    \              |
               |      \ (0,1)      |
               |        \          | 
               |(1,2)     \   (0,2)|
               |       (1,1)\      |
               |              \    |
               |   T1           \  |
               |       (1,0)      \|
            V2 +-------------------+ V0
               |       (2,0)      /|
               |                /  |
               |   T2         /    |
               |       (2,2)/      |
               |(2,1)     /   (3,0)|
               |        /          |
               |      / (3,1)      |
               |    /              |
               |  /           T3   |
               |/      (3,2)       |
            V4 +-------------------+ V5

        Triangle Connectivity and Sibling Half-Facet (Half-Edge) Data Struct:

        triangle |   vertices   |     sibling half-edges
         indices |  V0, V1, V2  |     E0,     E1,      E2
        ---------+--------------+-------------------------
            0    |   0,  3,  1  | (NULL),  (1,1),  (NULL)
            1    |   1,  2,  0  |  (2,0),  (0,1),  (NULL)
            2    |   4,  0,  2  |  (1,0), (NULL),   (3,1)
            3    |   4,  5,  0  | (NULL),  (2,2),  (NULL)
        """
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)

        cv = np.array([0, 3, 1, 1, 2, 0, 4, 0, 2, 4, 5, 0])
        cv.shape = (4,3)
        self.Cell.Append(cv)
        hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        hfs[1] = (1, 1)
        self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        hfs[0] = (2, 0)
        hfs[1] = (0, 1)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        hfs[0] = (1, 0)
        hfs[1] = NULL_HalfFacet
        hfs[2] = (3, 1)
        self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        hfs[0] = NULL_HalfFacet
        hfs[1] = (2, 2)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        self.Cell.Print()
        
        cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(2, 0)
        self.assertEqual(np.array_equal(cell_attach,np.array([])), True, "Should be True.")
        cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(0, 2)
        self.assertEqual(np.array_equal(cell_attach,np.array([2, 1, 0, 3])), True, "Should be True.")
        cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(2, 1)
        self.assertEqual(np.array_equal(cell_attach,np.array([1, 2])), True, "Should be True.")
        #print(cell_attach)
        
        self.Cell.Print_Two_Cells_Are_Facet_Connected(0, 1, 2)
        self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(0, 1, 2), True, "Should be True.")
        self.Cell.Print_Two_Cells_Are_Facet_Connected(0, 1, 3)
        self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(0, 1, 3), True, "Should be True.")
        self.Cell.Print_Two_Cells_Are_Facet_Connected(2, 1, 2)
        self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(2, 1, 2), True, "Should be True.")
        self.Cell.Print_Two_Cells_Are_Facet_Connected(2, 3, 2)
        self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(2, 3, 2), False, "Should be False.")
        print("")

        hf_in = np.array((1, 1), dtype=HalfFacetType)
        attached0 = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached0,np.array([(1,1), (0,1)], dtype=HalfFacetType)), True, "Should be True.")
        self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        hf_in[['ci','fi']] = (3, 1)
        attached1 = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached1,np.array([(3,1), (2,2)], dtype=HalfFacetType)), True, "Should be True.")
        self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        print("")

        free_bdy = self.Cell.Get_FreeBoundary()
        print("free boundary of the mesh:")
        print(free_bdy)
        free_bdy_CHK = np.full(6, NULL_HalfFacet, dtype=HalfFacetType)
        free_bdy_CHK[0] = (0, 0)
        free_bdy_CHK[1] = (0, 2)
        free_bdy_CHK[2] = (1, 2)
        free_bdy_CHK[3] = (2, 1)
        free_bdy_CHK[4] = (3, 0)
        free_bdy_CHK[5] = (3, 2)
        self.assertEqual(np.array_equal(free_bdy,free_bdy_CHK), True, "Should be True.")
        print("")

        non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        self.Cell.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([], dtype=HalfFacetType)), True, "Should be True.")

    def test_Manifold_1(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)

        # see Manifold_Triangle_Mesh_1.jpg
        cv = np.array([0,1,4, 1,2,4, 2,3,4, 3,0,4])
        cv.shape = (4,3)
        self.Cell.Append(cv)
        hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        hfs[0] = (1, 1)
        hfs[1] = (3, 0)
        self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        hfs[0] = (2, 1)
        hfs[1] = (0, 0)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        hfs[0] = (3, 1)
        hfs[1] = (1, 0)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        hfs[0] = (0, 1)
        hfs[1] = (2, 0)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        self.Cell.Print()

        attached_cl = self.Cell.Get_Cells_Attached_To_Vertex(2, 2)
        print(attached_cl)
        self.assertEqual(np.array_equal(attached_cl,np.array([2, 1], dtype=CellIndType)), True, "Should be [2, 1].")
        CHK_Facet_Connected = self.Cell.Two_Cells_Are_Facet_Connected(1, 0, 1)
        self.Cell.Print_Two_Cells_Are_Facet_Connected(1, 0, 1)
        self.assertEqual(CHK_Facet_Connected, True, "Should be True.")

        hf_in = np.array((3, 1), dtype=HalfFacetType)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        attached_hf_REF = np.full(2, NULL_HalfFacet, dtype=HalfFacetType)
        attached_hf_REF[0] = (3, 1)
        attached_hf_REF[1] = (2, 0)
        self.assertEqual(np.array_equal(attached_hf,attached_hf_REF), True, "Should be [(3,1), (2,0)].")

        hf_in[['ci','fi']] = (2,2)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached_hf,[hf_in]), True, "Should be (2,2).")

        non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        self.Cell.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([], dtype=HalfFacetType)), True, "Should be [].")

        uv = self.Cell.Get_Unique_Vertices()
        self.Cell.Print_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,np.array([0, 1, 2, 3, 4], dtype=VtxIndType)), True, "Should be [0, 1, 2, 3, 4].")

        EE = self.Cell.Get_Edges()
        self.Cell.Print_Edges()
        EE_ref = np.full(8, NULL_MeshEdge, dtype=MeshEdgeType)
        EE_ref[0] = (0, 1)
        EE_ref[1] = (0, 3)
        EE_ref[2] = (0, 4)
        EE_ref[3] = (1, 2)
        EE_ref[4] = (1, 4)
        EE_ref[5] = (2, 3)
        EE_ref[6] = (2, 4)
        EE_ref[7] = (3, 4)
        self.assertEqual(np.array_equal(EE,EE_ref), True, "Should be [(0, 1) (0, 3) (0, 4) (1, 2) (1, 4) (2, 3) (2, 4) (3, 4)].")

        FB = self.Cell.Get_FreeBoundary()
        print(FB)
        FB_ref = np.full(4, NULL_HalfFacet, dtype=HalfFacetType)
        FB_ref[0] = (0, 2)
        FB_ref[1] = (1, 2)
        FB_ref[2] = (2, 2)
        FB_ref[3] = (3, 2)
        self.assertEqual(np.array_equal(FB,FB_ref), True, "Should be [(0, 2) (1, 2) (2, 2) (3, 2)].")

    def test_Nonmanifold_1(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(3)

        # see Nonmanifold_Triangle_Mesh_1.jpg
        cv = np.array([0,1,2, 1,3,2, 1,4,2, 1,2,5])
        cv.shape = (4,3)
        self.Cell.Append(cv)
        hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        hfs[0] = (1, 1)
        self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        hfs[0] = NULL_HalfFacet
        hfs[1] = (2, 1)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        hfs[0] = NULL_HalfFacet
        hfs[1] = (3, 2)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        hfs[0] = NULL_HalfFacet
        hfs[1] = NULL_HalfFacet
        hfs[2] = (0, 0)
        self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        self.Cell.Print()

        attached_cl = self.Cell.Get_Cells_Attached_To_Vertex(2, 3)
        print(attached_cl)
        self.assertEqual(np.array_equal(attached_cl,np.array([3, 0, 1, 2], dtype=CellIndType)), True, "Should be [3, 0, 1, 2].")
        CHK_Facet_Connected = self.Cell.Two_Cells_Are_Facet_Connected(1, 0, 1)
        self.Cell.Print_Two_Cells_Are_Facet_Connected(1, 0, 1)
        self.assertEqual(CHK_Facet_Connected, True, "Should be True.")

        hf_in = np.array((1, 1), dtype=HalfFacetType)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        attached_hf_REF = np.full(4, NULL_HalfFacet, dtype=HalfFacetType)
        attached_hf_REF[0] = (1, 1)
        attached_hf_REF[1] = (2, 1)
        attached_hf_REF[2] = (3, 2)
        attached_hf_REF[3] = (0, 0)
        self.assertEqual(np.array_equal(attached_hf,attached_hf_REF), True, "Should be [(1,1), (2,1), (3,2), (0,0)].")

        hf_in[['ci','fi']] = (3,1)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached_hf,[hf_in]), True, "Should be (3,1).")

        non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        self.Cell.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([(3, 2)], dtype=HalfFacetType)), True, "Should be [(3, 2)].")

        uv = self.Cell.Get_Unique_Vertices()
        self.Cell.Print_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,np.array([0, 1, 2, 3, 4, 5], dtype=VtxIndType)), True, "Should be [0, 1, 2, 3, 4, 5].")

        EE = self.Cell.Get_Edges()
        self.Cell.Print_Edges()
        EE_ref = np.full(9, NULL_MeshEdge, dtype=MeshEdgeType)
        EE_ref[0] = (0, 1)
        EE_ref[1] = (0, 2)
        EE_ref[2] = (1, 2)
        EE_ref[3] = (1, 3)
        EE_ref[4] = (1, 4)
        EE_ref[5] = (1, 5)
        EE_ref[6] = (2, 3)
        EE_ref[7] = (2, 4)
        EE_ref[8] = (2, 5)
        self.assertEqual(np.array_equal(EE,EE_ref), True, "Should be [(0, 1) (0, 2) (1, 2) (1, 3) (1, 4) (1, 5) (2, 3) (2, 4) (2, 5)].")

        FB = self.Cell.Get_FreeBoundary()
        print(FB)
        FB_ref = np.full(8, NULL_HalfFacet, dtype=HalfFacetType)
        FB_ref[0] = (0, 1)
        FB_ref[1] = (0, 2)
        FB_ref[2] = (1, 0)
        FB_ref[3] = (1, 2)
        FB_ref[4] = (2, 0)
        FB_ref[5] = (2, 2)
        FB_ref[6] = (3, 0)
        FB_ref[7] = (3, 1)
        self.assertEqual(np.array_equal(FB,FB_ref), True, "Should be [(0, 1) (0, 2) (1, 0) (1, 2) (2, 0) (2, 2) (3, 0) (3, 1)].")

    def test_Nonmanifold_2(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(3)

        # see Nonmanifold_Triangle_Mesh_2.jpg
        cv = np.array([0,1,5, 5,2,1, 1,5,3, 4,5,1, 9,6,5, 9,5,7, 8,9,5])
        cv.shape = (7,3)
        self.Cell.Append(cv)
        hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        hfs[0] = (1, 1)
        self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        hfs[0] = NULL_HalfFacet
        hfs[1] = (2, 2)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        hfs[0] = NULL_HalfFacet
        hfs[1] = NULL_HalfFacet
        hfs[2] = (3, 0)
        self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        hfs[0] = (0, 0)
        hfs[1] = NULL_HalfFacet
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        hfs[0] = NULL_HalfFacet
        hfs[1] = (5, 2)
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[4][:] = hfs[:] # Cell #4
        hfs[0] = NULL_HalfFacet
        hfs[1] = NULL_HalfFacet
        hfs[2] = (6, 0)
        self.Cell.halffacet[5][:] = hfs[:] # Cell #5
        hfs[0] = (4, 1)
        hfs[1] = NULL_HalfFacet
        hfs[2] = NULL_HalfFacet
        self.Cell.halffacet[6][:] = hfs[:] # Cell #6
        self.Cell.Print()

        attached_cl_1 = self.Cell.Get_Cells_Attached_To_Vertex(5, 0)
        print(attached_cl_1)
        self.assertEqual(np.array_equal(attached_cl_1,np.array([0, 1, 2, 3], dtype=CellIndType)), True, "Should be [0, 1, 2, 3].")
        attached_cl_2 = self.Cell.Get_Cells_Attached_To_Vertex(5, 4)
        print(attached_cl_2)
        self.assertEqual(np.array_equal(attached_cl_2,np.array([4, 5, 6], dtype=CellIndType)), True, "Should be [4, 5, 6].")

        CHK_Facet_Connected = self.Cell.Two_Cells_Are_Facet_Connected(5, 0, 4)
        self.Cell.Print_Two_Cells_Are_Facet_Connected(5, 0, 4)
        self.assertEqual(CHK_Facet_Connected, False, "Should be False.")

        hf_in = np.array((1, 1), dtype=HalfFacetType)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        attached_hf_REF = np.full(4, NULL_HalfFacet, dtype=HalfFacetType)
        attached_hf_REF[0] = (1, 1)
        attached_hf_REF[1] = (2, 2)
        attached_hf_REF[2] = (3, 0)
        attached_hf_REF[3] = (0, 0)
        self.assertEqual(np.array_equal(attached_hf,attached_hf_REF), True, "Should be [(1,1), (2,2), (3,0), (0,0)].")

        hf_in[['ci','fi']] = (4,2)
        attached_hf = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached_hf,[hf_in]), True, "Should be (4,2).")

        non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        self.Cell.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([(3, 0), (6, 0)], dtype=HalfFacetType)), True, "Should be [(3, 0), (6, 0)].")

        uv = self.Cell.Get_Unique_Vertices()
        self.Cell.Print_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=VtxIndType)), True, "Should be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].")

        EE = self.Cell.Get_Edges()
        self.Cell.Print_Edges()
        EE_ref = np.full(16, NULL_MeshEdge, dtype=MeshEdgeType)
        EE_ref[0]  = (0, 1)
        EE_ref[1]  = (0, 5)
        EE_ref[2]  = (1, 2)
        EE_ref[3]  = (1, 3)
        EE_ref[4]  = (1, 4)
        EE_ref[5]  = (1, 5)
        EE_ref[6]  = (2, 5)
        EE_ref[7]  = (3, 5)
        EE_ref[8]  = (4, 5)
        EE_ref[9]  = (5, 6)
        EE_ref[10] = (5, 7)
        EE_ref[11] = (5, 8)
        EE_ref[12] = (5, 9)
        EE_ref[13] = (6, 9)
        EE_ref[14] = (7, 9)
        EE_ref[15] = (8, 9)
        self.assertEqual(np.array_equal(EE,EE_ref), True, "Should be [(0, 1) (0, 5) (1, 2) (1, 3) (1, 4) (1, 5) (2, 5) (3, 5) (4, 5) (5, 6) (5, 7) (5, 8) (5, 9) (6, 9) (7, 9) (8, 9)].")

        FB = self.Cell.Get_FreeBoundary()
        print(FB)
        FB_ref = np.full(14, NULL_HalfFacet, dtype=HalfFacetType)
        FB_ref[0]  = (0, 1)
        FB_ref[1]  = (0, 2)
        FB_ref[2]  = (1, 0)
        FB_ref[3]  = (1, 2)
        FB_ref[4]  = (2, 0)
        FB_ref[5]  = (2, 1)
        FB_ref[6]  = (3, 1)
        FB_ref[7]  = (3, 2)
        FB_ref[8]  = (4, 0)
        FB_ref[9]  = (4, 2)
        FB_ref[10] = (5, 0)
        FB_ref[11] = (5, 1)
        FB_ref[12] = (6, 1)
        FB_ref[13] = (6, 2)
        self.assertEqual(np.array_equal(FB,FB_ref), True, "Should be [(0, 1) (0, 2) (1, 0) (1, 2) (2, 0) (2, 1) (3, 1) (3, 2) (4, 0) (4, 2) (5, 0) (5, 1) (6, 1) (6, 2)].")

    def test_Print(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)

        cell_data = np.array([1, 6, 2, 18, 13, 9, 34, 12, 43, 7, 12, 16, 99, 39, 19])
        cell_data.shape = (5,3)
        self.Cell.Set(cell_data)
        print(" ")
        self.Cell.Print()
        
        unique_vertices = self.Cell.Get_Unique_Vertices()
        self.assertEqual( np.array_equal(unique_vertices,[1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99]), True, "Should be [1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99].")
        self.Cell.Print_Unique_Vertices()

        self.VC.Init_Coord(3)
        self.assertEqual(np.array_equal(self.VC.coord[1],[0.0, 0.0]), True, "Should be [0.0, 0.0].")

        coord_data = np.array([7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, -19.4, 12.6, 35.7, -66.7, 32.8])
        coord_data.shape = (6,2)
        self.VC.Set(coord_data)
        self.assertEqual(np.array_equal(self.VC.coord[3],[10.1, -19.4]), True, "Should be [10.1, -19.4].")
        print(" ")
        self.VC.Print()
        self.VC.Print(3, num_digit=10)

    def test_MeshEdgeType(self):
        del(self.Cell)
        del(self.VC)
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

    def test_Get_Edges(self):
        del(self.Cell)
        del(self.VC)
        self.Cell = CellSimplexType(2)
        self.VC   = VtxCoordType(2)
        print(" ")

        cv = np.array([0, 3, 5, 4, 12, 9, 88, 62, 54, 99, 120, 101])
        cv.shape = (4,3)
        self.Cell.Append(cv)
        self.Cell.Print()
        EE = self.Cell.Get_Edges()
        self.Cell.Print_Edges()
        
        Edge_CHK = np.full(12, NULL_MeshEdge, dtype=MeshEdgeType)
        Edge_CHK[0]  = (0, 3)
        Edge_CHK[1]  = (0, 5)
        Edge_CHK[2]  = (3, 5)
        Edge_CHK[3]  = (4, 9)
        Edge_CHK[4]  = (4, 12)
        Edge_CHK[5]  = (9, 12)
        Edge_CHK[6]  = (54, 62)
        Edge_CHK[7]  = (54, 88)
        Edge_CHK[8]  = (62, 88)
        Edge_CHK[9]  = (99, 101)
        Edge_CHK[10] = (99, 120)
        Edge_CHK[11] = (101, 120)
        self.assertEqual(np.array_equal(Edge_CHK,EE), True, "Should be True.")

if __name__ == '__main__':
    unittest.main()
