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
        print(" ")

    def test_Size_Capacity_Dim(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(2)
        
        cell_vtx = np.array([0, 3, 1, 1, 2, 0, 4, 0, 2, 4, 5, 0])
        cell_vtx.shape = (4,3)
        self.Mesh.Append_Cell(cell_vtx)
        print(self.Mesh)

        self.assertEqual(self.Mesh.Num_Cell(), 4, "Num_Cell should be 4.")
        self.assertEqual(self.Mesh.Top_Dim(), 2, "Top_Dim should be 2.")
        
        Cell_cap, Vtx2HF_cap = self.Mesh.Capacity()
        self.assertEqual(Cell_cap, 5, "Cell Capacity should be 5.")
        self.assertEqual(Vtx2HF_cap, 0, "Vtx2HF capacity should be 0.")

    def test_Reserve(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(4)

        num_reserve = 4
        self.Mesh.Reserve(num_reserve)
        Cell_cap, Vtx2HalfFacets_cap = self.Mesh.Capacity()
        print(self.Mesh)
        self.assertEqual(Cell_cap, num_reserve+1, "Cell capacity should be " + str(num_reserve+1) + ".")
        self.assertEqual(Vtx2HalfFacets_cap, 0, "Vtx2HalfFacets capacity should be " + str(0) + ".")
        _v2hfs_cap = np.rint((4+1) * num_reserve * 1.2)
        self.assertEqual(self.Mesh._v2hfs.Capacity(), _v2hfs_cap, "_v2hfs capacity should be " + str(_v2hfs_cap) + ".")
        
    def test_Append_and_Set(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(3)

        self.Mesh.Reserve(5)
        print(" ")
        
        self.Mesh.Append_Cell(np.array([1, 2, 3, 4]))
        self.Mesh.Append_Cell(np.array([4, 23, 88]))
        print(self.Mesh)
        self.Mesh.Cell.Print()

        new_cell_vtx = np.array([2, 5, 4, 11, 6, 88, 9, 13, 1, 4, 90, 74, 23, 45, 71, 55])
        new_cell_vtx.shape = (4,4)
        self.Mesh.Append_Cell(new_cell_vtx)
        
        self.Mesh.Cell.Print()
        
        self.Mesh.Set_Cell(3, np.array([57, 14, 33, 48]))
        self.Mesh.Cell.Print()

        self.assertEqual(np.array_equal(self.Mesh.Cell.vtx[2],[6, 88, 9, 13]), True, "Should be [6, 88, 9, 13].")
        self.assertEqual(self.Mesh.Cell.Size(), 5, "Should be 5.")
        
        uv = self.Mesh.Get_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,[1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88]), True, "Should be [1, 2, 3, 4, 5, 6, 9, 11, 13, 14, 23, 33, 45, 48, 55, 57, 71, 88].")
        
        Num_Vtx = self.Mesh.Num_Vtx()
        self.assertEqual(Num_Vtx, 18, "Should be 18.")
        Max_vi = self.Mesh.Max_Vtx_Index()
        self.assertEqual(Max_vi, 88, "Should be 88.")

    def test_2D_Manifold_Mesh_1(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(2)

        # see Manifold_Triangle_Mesh_1.jpg
        cv = np.array([0,1,4, 1,2,4, 2,3,4, 3,0,4])
        cv.shape = (4,3)
        self.Mesh.Append_Cell(cv)

        self.Mesh._Finalize_v2hfs(True)
        self.Mesh.Print_v2hfs()

        # check '_v2hfs' against reference data
        vhfs_REF = np.full(12, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        vhfs_REF[0]  = (1, 0, 2)
        vhfs_REF[1]  = (2, 1, 2)
        vhfs_REF[2]  = (3, 2, 2)
        vhfs_REF[3]  = (3, 3, 2)
        vhfs_REF[4]  = (4, 0, 0)
        vhfs_REF[5]  = (4, 0, 1)
        vhfs_REF[6]  = (4, 1, 0)
        vhfs_REF[7]  = (4, 1, 1)
        vhfs_REF[8]  = (4, 2, 0)
        vhfs_REF[9]  = (4, 2, 1)
        vhfs_REF[10] = (4, 3, 0)
        vhfs_REF[11] = (4, 3, 1)
        #print(self.Mesh._v2hfs.VtxMap[0:self.Mesh._v2hfs.Size()])
        #print(vhfs_REF)
        self.assertEqual(np.array_equal(self.Mesh._v2hfs.VtxMap[0:self.Mesh._v2hfs.Size()],vhfs_REF), True, "Should be True.")

        # fill out the sibling half-facet data in 'self.Mesh.Cell'
        self.Mesh._Build_Sibling_HalfFacets()
        self.Mesh.Cell.Print()
        
        # check 'Cell.halffacet' against reference data
        Cell_HF_REF = np.full((4,3), NULL_HalfFacet, dtype=HalfFacetType)
        Cell_HF_REF[0][0] = (1, 1)
        Cell_HF_REF[0][1] = (3, 0)
        Cell_HF_REF[1][0] = (2, 1)
        Cell_HF_REF[1][1] = (0, 0)
        Cell_HF_REF[2][0] = (3, 1)
        Cell_HF_REF[2][1] = (1, 0)
        Cell_HF_REF[3][0] = (0, 1)
        Cell_HF_REF[3][1] = (2, 0)
        #print(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()])
        #print(Cell_HF_REF)
        self.assertEqual(np.array_equal(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()],Cell_HF_REF), True, "Should be True.")

        # generate the Vtx2HalfFacets data struct
        self.Mesh._Build_Vtx2HalfFacets()
        self.Mesh.Close() # we can close it now, i.e. no more modifications
        self.Mesh.Print_Vtx2HalfFacets()

        # check 'Vtx2HalfFacets' against reference data
        Vtx2HalfFacets_REF = np.full(5, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        Vtx2HalfFacets_REF[0]  = (0, 3, 2)
        Vtx2HalfFacets_REF[1]  = (1, 1, 2)
        Vtx2HalfFacets_REF[2]  = (2, 2, 2)
        Vtx2HalfFacets_REF[3]  = (3, 3, 2)
        Vtx2HalfFacets_REF[4]  = (4, 0, 0)
        #print(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()])
        #print(Vtx2HalfFacets_REF)
        self.assertEqual(np.array_equal(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()],\
                         Vtx2HalfFacets_REF), True, "Should be True.")




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
