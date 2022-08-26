import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.BaseSimplexMesh import *

class TestBaseSimplexMesh(unittest.TestCase):

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

    def test_Size_and_Dim(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(2)
        
        cell_vtx = np.array([0, 3, 1, 1, 2, 0, 4, 0, 2, 4, 5, 0])
        cell_vtx.shape = (4,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        self.assertEqual(self.Mesh.Num_Cell(), 4, "Num_Cell should be 4.")
        self.assertEqual(self.Mesh.Top_Dim(), 2, "Top_Dim should be 2.")

    # def test_Reserve(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(4)
        # self.VC   = VtxCoordType(6)

        # num_reserve = 4
        # self.Cell.Reserve(num_reserve)
        # self.VC.Reserve(num_reserve)
        # self.assertEqual(self.Cell.vtx.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")
        # self.assertEqual(self.Cell.halffacet.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")
        # self.assertEqual(self.VC.coord.shape[0], num_reserve, "Reserved size should be " + str(num_reserve) + ".")

    # def test_Append_and_Set(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(3)
        # self.VC   = VtxCoordType(3)

        # self.Cell.Reserve(5)
        # print(" ")
        
        # self.Cell.Append([1, 2, 3, 4])
        # self.Cell.Append([4, 23, 88])
        # print(self.Cell)

        # new_cell_vtx = [2, 5, 4, 11, 6, 88, 9, 13, 1, 4, 90, 74, 23, 45, 71, 55]
        # self.Cell.Append_Batch(4, new_cell_vtx)

        # print(self.Cell.vtx)
        # print(self.Cell)

        # self.Cell.Set(3, [57, 14, 33, 48])
        # print(self.Cell.vtx)

        # self.assertEqual(np.array_equal(self.Cell.vtx[2],[6, 88, 9, 13]), True, "Should be [6, 88, 9, 13].")
        # self.assertEqual(self.Cell.Size(), 5, "Should be 5.")
        
        # uv = self.Cell.Get_Unique_Vertices()
        # self.assertEqual(np.array_equal(uv,[1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88]), True, "Should be [1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88].")
        
        # Num_Vtx = self.Cell.Num_Vtx()
        # self.assertEqual(Num_Vtx, 18, "Should be 18.")
        # Max_vi = self.Cell.Max_Vtx_Index()
        # self.assertEqual(Max_vi, 88, "Should be 88.")
        
        # self.VC.Reserve(5)
        # print(" ")
        
        # self.VC.Append([0.2, 1.4, -9.4])
        # self.VC.Append([4.4, 2.3])
        # print(self.VC)

        # new_vtx_coord = [3.1, -2.5, 7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, 4.7, 10.6, -5.1]
        # self.VC.Append_Batch(4, new_vtx_coord)

        # print(self.VC.coord)
        # print(self.VC)

        # self.VC.Set(3, [8.8, -6.7, 2.7])
        # print(self.VC.coord)

        # self.assertEqual(np.array_equal(self.VC.coord[2],[-1.2, 4.4, 6.6]), True, "Should be [-1.2, 4.4, 6.6].")
        # self.assertEqual(np.array_equal(self.VC.coord[4],[4.7, 10.6, -5.1]), True, "Should be [4.7, 10.6, -5.1].")
        # self.assertEqual(self.VC.Size(), 5, "Should be 5.")

    # def test_Reindex(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(3)
        # self.VC   = VtxCoordType(3)

        # new_cell_vtx = [0, 3, 5, 1, 4, 12, 9, 6, 88, 62, 54, 72, 99, 120, 101, 154]
        # self.Cell.Append_Batch(4, new_cell_vtx)
        # self.Cell.Print()
        
        # new_indices = np.zeros(155)
        # lin_indices = np.linspace(0, 15, num=16, dtype=VtxIndType)
        # new_indices[new_cell_vtx] = lin_indices
        # # print(new_indices)
        # # print(lin_indices)
        # self.Cell.Reindex_Vertices(new_indices)
        # self.Cell.Print()
        # CHK_Cell_vtx = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        # self.assertEqual(np.array_equal(self.Cell.vtx,CHK_Cell_vtx), True, "Should be [[ 0  1  2  3], [ 4  5  6  7], [ 8  9 10 11], [12 13 14 15]].")

    # def test_Baby_Methods(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(3)
        # self.VC   = VtxCoordType(3)
        
        # Cell_0D = CellSimplexType(0)
        # Cell_0D.Append_Batch(4, [88, 4, 21, 14])
        # print(Cell_0D)
        # self.assertEqual(Cell_0D.Adj_Vertices_In_Facet_Equal(np.array([]), np.array([])), True, "Should be True.")
        # self.assertEqual(Cell_0D.Get_Vertex_With_Largest_Index_In_Facet(np.array([88]), 0)==NULL_Vtx, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_0D.Get_Adj_Vertices_In_Facet(np.array([]), 34),np.array([])), True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_0D.Get_Global_Vertices_In_Facet(np.array([88]), 0),np.array([])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_0D.Get_Local_Facets_Sharing_Local_Vertex(0),np.array([])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_0D.Get_Local_Vertices_Of_Local_Facet(0),np.array([])), \
                         # True, "Should be True.")
        # self.assertEqual(Cell_0D.Get_Local_Vertex_Index_In_Cell(188, [32])==NULL_Small, True, "Should be True.")
        # self.assertEqual(Cell_0D.Get_Local_Vertex_Index_In_Cell(32, [32])==0, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_0D.Vtx2Adjacent(5, 2, 0),np.array([])), True, "Should be True.")

        # Cell_1D = CellSimplexType(1)
        # Cell_1D.Append_Batch(4, [0, 3, 4, 12, 88, 62, 99, 120])
        # print(Cell_1D)
        # self.assertEqual(Cell_1D.Adj_Vertices_In_Facet_Equal(np.array([]), np.array([])), True, "Should be True.")
        # self.assertEqual(Cell_1D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35]), 1)==78, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_1D.Get_Adj_Vertices_In_Facet(np.array([67]), 34),np.array([])), True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_1D.Get_Global_Vertices_In_Facet(np.array([23, 34]), 0),np.array([34])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_1D.Get_Local_Facets_Sharing_Local_Vertex(1),np.array([0])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_1D.Get_Local_Vertices_Of_Local_Facet(0),np.array([1])), \
                         # True, "Should be True.")
        # self.assertEqual(Cell_1D.Get_Local_Vertex_Index_In_Cell(188, [19, 56])==NULL_Small, True, "Should be True.")
        # self.assertEqual(Cell_1D.Get_Local_Vertex_Index_In_Cell(56, [19, 56])==1, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_1D.Vtx2Adjacent(87, 2, 1),np.array([])), True, "Should be True.")
        
        # Cell_2D = CellSimplexType(2)
        # Cell_2D.Append_Batch(4, [0, 3, 5, 4, 12, 9, 88, 62, 54, 99, 120, 101])
        # print(Cell_2D)
        # self.assertEqual(Cell_2D.Adj_Vertices_In_Facet_Equal(np.array([12]), np.array([13])), False, "Should be False.")
        # self.assertEqual(Cell_2D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 66]), 0)==66, True, "Should be True.")
        # self.assertEqual(Cell_2D.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 66]), 1)==78, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Get_Adj_Vertices_In_Facet(np.array([18, 91]), 34),np.array([NULL_Vtx])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Get_Adj_Vertices_In_Facet(np.array([18, 91]), 18),np.array([91])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Get_Global_Vertices_In_Facet(np.array([53, 34, 12]), 1),np.array([53, 12])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Get_Local_Facets_Sharing_Local_Vertex(2),np.array([0, 1])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Get_Local_Vertices_Of_Local_Facet(1),np.array([0, 2])), \
                         # True, "Should be True.")
        # self.assertEqual(Cell_2D.Get_Local_Vertex_Index_In_Cell(188, [91, 29, 56])==NULL_Small, True, "Should be True.")
        # self.assertEqual(Cell_2D.Get_Local_Vertex_Index_In_Cell(29, [91, 29, 56])==1, True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(87, 3, 1),np.array([NULL_Vtx])), True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(101, 3, 1),np.array([99])), True, "Should be True.")
        # self.assertEqual(np.array_equal(Cell_2D.Vtx2Adjacent(99, 3, 2),np.array([120])), True, "Should be True.")
        
        # self.Cell.Reserve(5)
        # print(" ")
        
        # self.Cell.Append_Batch(4, [0, 3, 5, 1, 4, 12, 9, 6, 88, 62, 54, 72, 99, 120, 101, 154])
        # print(self.Cell)
        # self.assertEqual(self.Cell.Adj_Vertices_In_Facet_Equal(np.array([12, 34]), np.array([13, 34])), False, "Should be False.")
        # self.assertEqual(self.Cell.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 17, 66]), 2)==78, True, "Should be True.")
        # self.assertEqual(self.Cell.Get_Vertex_With_Largest_Index_In_Facet(np.array([78, 35, 17, 66]), 0)==66, True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Get_Adj_Vertices_In_Facet(np.array([18, 91, 23]), 34), \
                         # np.array([NULL_Vtx, NULL_Vtx])), True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Get_Adj_Vertices_In_Facet(np.array([18, 91, 23]), 18), \
                         # np.array([91, 23])), True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Get_Global_Vertices_In_Facet(np.array([53, 5, 12, 8]), 2),np.array([53, 5, 8])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Get_Local_Facets_Sharing_Local_Vertex(3),np.array([0, 1, 2])), \
                         # True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Get_Local_Vertices_Of_Local_Facet(1),np.array([0, 2, 3])), \
                         # True, "Should be True.")
        # self.assertEqual(self.Cell.Get_Local_Vertex_Index_In_Cell(188, [91, 29, 13, 56])==NULL_Small, True, "Should be True.")
        # self.assertEqual(self.Cell.Get_Local_Vertex_Index_In_Cell(13, [91, 29, 13, 56])==2, True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(87, 1, 2),np.array([NULL_Vtx, NULL_Vtx])), True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(9, 1, 1),np.array([4, 6])), True, "Should be True.")
        # self.assertEqual(np.array_equal(self.Cell.Vtx2Adjacent(4, 1, 2),np.array([12, 6])), True, "Should be True.")

    # def test_Attached_and_Connected_and_FreeBdy(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(2)

        # self.Cell.Append_Batch(4, [0, 3, 1, 1, 2, 0, 4, 0, 2, 4, 5, 0])
        # hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        # hfs[1] = (1, 1)
        # self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        # hfs[0] = (2, 0)
        # hfs[1] = (0, 1)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        # hfs[0] = (1, 0)
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = (3, 1)
        # self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = (2, 2)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        # self.Cell.Print()
        
        # cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(2, 0)
        # self.assertEqual(np.array_equal(cell_attach,np.array([])), True, "Should be True.")
        # cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(0, 2)
        # self.assertEqual(np.array_equal(cell_attach,np.array([2, 1, 0, 3])), True, "Should be True.")
        # cell_attach = self.Cell.Get_Cells_Attached_To_Vertex(2, 1)
        # self.assertEqual(np.array_equal(cell_attach,np.array([1, 2])), True, "Should be True.")
        # #print(cell_attach)
        
        # self.Cell.Print_Two_Cells_Are_Facet_Connected(0, 1, 2)
        # self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(0, 1, 2), True, "Should be True.")
        # self.Cell.Print_Two_Cells_Are_Facet_Connected(0, 1, 3)
        # self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(0, 1, 3), True, "Should be True.")
        # self.Cell.Print_Two_Cells_Are_Facet_Connected(2, 1, 2)
        # self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(2, 1, 2), True, "Should be True.")
        # self.Cell.Print_Two_Cells_Are_Facet_Connected(2, 3, 2)
        # self.assertEqual(self.Cell.Two_Cells_Are_Facet_Connected(2, 3, 2), False, "Should be False.")
        # print("")

        # hf_in = np.array((1, 1), dtype=HalfFacetType)
        # attached0 = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        # self.assertEqual(np.array_equal(attached0,np.array([(1,1), (0,1)], dtype=HalfFacetType)), True, "Should be True.")
        # self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        # hf_in[['ci','fi']] = (3, 1)
        # attached1 = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        # self.assertEqual(np.array_equal(attached1,np.array([(3,1), (2,2)], dtype=HalfFacetType)), True, "Should be True.")
        # self.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        # print("")

        # free_bdy = self.Cell.Get_FreeBoundary()
        # print("free boundary of the mesh:")
        # print(free_bdy)
        # free_bdy_CHK = np.full(6, NULL_HalfFacet, dtype=HalfFacetType)
        # free_bdy_CHK[0] = (0, 0)
        # free_bdy_CHK[1] = (0, 2)
        # free_bdy_CHK[2] = (1, 2)
        # free_bdy_CHK[3] = (2, 1)
        # free_bdy_CHK[4] = (3, 0)
        # free_bdy_CHK[5] = (3, 2)
        # self.assertEqual(np.array_equal(free_bdy,free_bdy_CHK), True, "Should be True.")
        # print("")

        # non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        # self.Cell.Print_Nonmanifold_HalfFacets()
        # self.assertEqual(np.array_equal(non_man_hf,np.array([], dtype=HalfFacetType)), True, "Should be True.")
        # print("")

    # def test_Nonmanifold_1(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(3)

        # # see Nonmanifold_Triangle_Mesh_1.jpg
        # self.Cell.Append_Batch(4, [0,1,2, 1,3,2, 1,4,2, 1,2,5])
        # hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        # hfs[0] = (1, 1)
        # self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = (2, 1)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = (3, 2)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = (0, 0)
        # self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        # self.Cell.Print()

        # non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        # self.Cell.Print_Nonmanifold_HalfFacets()
        # self.assertEqual(np.array_equal(non_man_hf,np.array([(3, 2)], dtype=HalfFacetType)), True, "Should be True.")
        # print("")

    # def test_Nonmanifold_2(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(3)

        # # see Nonmanifold_Triangle_Mesh_2.jpg
        # self.Cell.Append_Batch(7, [0,1,5, 5,2,1, 1,5,3, 4,5,1, 9,6,5, 9,5,7, 8,9,5])
        # hfs = np.full(3, NULL_HalfFacet, dtype=HalfFacetType)
        # hfs[0] = (1, 1)
        # self.Cell.halffacet[0][:] = hfs[:] # Cell #0
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = (2, 2)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[1][:] = hfs[:] # Cell #1
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = (3, 0)
        # self.Cell.halffacet[2][:] = hfs[:] # Cell #2
        # hfs[0] = (0, 0)
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[3][:] = hfs[:] # Cell #3
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = (5, 2)
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[4][:] = hfs[:] # Cell #4
        # hfs[0] = NULL_HalfFacet
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = (6, 0)
        # self.Cell.halffacet[5][:] = hfs[:] # Cell #5
        # hfs[0] = (4, 1)
        # hfs[1] = NULL_HalfFacet
        # hfs[2] = NULL_HalfFacet
        # self.Cell.halffacet[6][:] = hfs[:] # Cell #6
        # self.Cell.Print()

        # non_man_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        # self.Cell.Print_Nonmanifold_HalfFacets()
        # self.assertEqual(np.array_equal(non_man_hf,np.array([(3, 0), (6, 0)], dtype=HalfFacetType)), True, "Should be True.")
        # print("")

    # def test_Print(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(2)

        # cell_data = [1, 6, 2, 18, 13, 9, 34, 12, 43, 7, 12, 16, 99, 39, 19]
        # self.Cell.Set_All(5, cell_data)
        # print(" ")
        # self.Cell.Print()
        
        # unique_vertices = self.Cell.Get_Unique_Vertices()
        # self.assertEqual( np.array_equal(unique_vertices,[1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99]), True, "Should be [1, 2, 6, 7, 9, 12, 13, 16, 18, 19, 34, 39, 43, 99].")
        # self.Cell.Print_Unique_Vertices()

        # coord_data = [7.3, -1.2, 4.4, 6.6, 3.6, -7.8, 10.1, -19.4, 12.6, 35.7, -66.7, 32.8]
        # self.VC.Set_All(6, coord_data)
        # self.assertEqual(np.array_equal(self.VC.coord[3],[10.1, -19.4]), True, "Should be [10.1, -19.4].")
        # print(" ")
        # self.VC.Print()

    # def test_MeshEdgeType(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(2)
        # print(" ")

        # Edge_array = np.full(6, NULL_MeshEdge, dtype=MeshEdgeType)
        # Edge_array[0] = (12, 14)
        # Edge_array[1] = (1, 5)
        # Edge_array[2] = (4, 23)
        # Edge_array[3] = (3, 11)
        # Edge_array[4] = (34, 16)
        # Edge_array[5] = (45, 28)

        # print(Edge_array)
        
        # EE2 = np.array((4, 23), dtype=MeshEdgeType)
        # self.assertEqual(Edge_array[2]==EE2, True, "Should be (4, 23).")
        # self.assertEqual(Edge_array[4]['v1']==16, True, "Should be 16.")

    # def test_Get_Edges(self):
        # del(self.Cell)
        # del(self.VC)
        # self.Cell = CellSimplexType(2)
        # self.VC   = VtxCoordType(2)
        # print(" ")

        # self.Cell.Append_Batch(4, [0, 3, 5, 4, 12, 9, 88, 62, 54, 99, 120, 101])
        # self.Cell.Print()
        # EE = self.Cell.Get_Edges()
        # self.Cell.Print_Edges()
        
        # Edge_CHK = np.full(12, NULL_MeshEdge, dtype=MeshEdgeType)
        # Edge_CHK[0]  = (0, 3)
        # Edge_CHK[1]  = (0, 5)
        # Edge_CHK[2]  = (3, 5)
        # Edge_CHK[3]  = (4, 9)
        # Edge_CHK[4]  = (4, 12)
        # Edge_CHK[5]  = (9, 12)
        # Edge_CHK[6]  = (54, 62)
        # Edge_CHK[7]  = (54, 88)
        # Edge_CHK[8]  = (62, 88)
        # Edge_CHK[9]  = (99, 101)
        # Edge_CHK[10] = (99, 120)
        # Edge_CHK[11] = (101, 120)
        # self.assertEqual(np.array_equal(Edge_CHK,EE), True, "Should be True.")

if __name__ == '__main__':
    unittest.main()
