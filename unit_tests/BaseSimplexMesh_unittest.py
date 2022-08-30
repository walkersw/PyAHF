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

    def test_Reindex(self):
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(3)

        cv = np.array([0,3,5,1, 4,12,9,6, 88,62,54,72, 99,120,101,154])
        cv.shape = (4,4)
        self.Mesh.Append_Cell(cv)

        self.Mesh.Finalize_Mesh_Connectivity()
        self.Mesh.Cell.Print()
        self.Mesh.Print_Vtx2HalfFacets()

        new_indices = np.zeros(155)
        lin_indices = np.linspace(0, 15, num=16, dtype=VtxIndType)
        cv.shape = (16,)
        new_indices[cv] = lin_indices
        # print(new_indices)
        # print(lin_indices)
        self.Mesh.Open()
        self.Mesh.Reindex_Vertices(new_indices)
        self.Mesh.Print()
        CHK_Cell_vtx = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        self.assertEqual(np.array_equal(self.Mesh.Cell.vtx[0:4],CHK_Cell_vtx), True, "Should be [[ 0  1  2  3], [ 4  5  6  7], [ 8  9 10 11], [12 13 14 15]].")

        # check 'Vtx2HalfFacets' against reference data
        Vtx2HalfFacets_REF = np.full(16, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        Vtx2HalfFacets_REF[0]  = (0,  0, 3)
        Vtx2HalfFacets_REF[1]  = (1,  0, 3)
        Vtx2HalfFacets_REF[2]  = (2,  0, 3)
        Vtx2HalfFacets_REF[3]  = (3,  0, 2)
        Vtx2HalfFacets_REF[4]  = (4,  1, 3)
        Vtx2HalfFacets_REF[5]  = (5,  1, 3)
        Vtx2HalfFacets_REF[6]  = (6,  1, 3)
        Vtx2HalfFacets_REF[7]  = (7,  1, 2)
        Vtx2HalfFacets_REF[8]  = (8,  2, 3)
        Vtx2HalfFacets_REF[9]  = (9,  2, 3)
        Vtx2HalfFacets_REF[10] = (10, 2, 3)
        Vtx2HalfFacets_REF[11] = (11, 2, 2)
        Vtx2HalfFacets_REF[12] = (12, 3, 3)
        Vtx2HalfFacets_REF[13] = (13, 3, 3)
        Vtx2HalfFacets_REF[14] = (14, 3, 3)
        Vtx2HalfFacets_REF[15] = (15, 3, 2)
        #print(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()])
        #print(Vtx2HalfFacets_REF)
        self.assertEqual(np.array_equal(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()],\
                         Vtx2HalfFacets_REF), True, "Should be True.")

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

    def test_0D_Nonmanifold_Mesh_1(self):
        """
        We create a mesh of 3 vertices with 0-D cells. Embedding it into \R^1, it looks like:

                 +         ++          +
               V0(C0)    V1(C1,C2)    V2(C3)

        where the Vi are the vertex indices, and Ci are the cell indices.

        Note: since the dimension is ZERO, the notion of manifold does not make sense.
        Moreover, note that in 0-D, cells, half-facets, and vertices are (topologically)
        the same; but in higher dimensions, this is not true.

        There are no non-manifold half-facets in 0-D.  But we treat V1, here, as a
        non-manifold vertex, because it is connected to two cells.  I.e. a vertex and
        cell should be uniquely identifiable (locally) for a 0-dimensional "manifold".
        This is done for the sake of generality with higher dimensions.
        
        The PyAHF code treats non-manifold vertices as *different* from non-manifold
        half-facets in all dimensions.
        """
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(0)

        # see help text above
        cv = np.array([0, 1, 1, 2])
        cv.shape = (4,1)
        self.Mesh.Append_Cell(cv)

        self.Mesh.Finalize_Mesh_Connectivity()
        self.Mesh.Cell.Print()
        self.Mesh.Print_Vtx2HalfFacets()

        # check 'Cell.halffacet' against reference data
        Cell_HF_REF = np.full((4,1), NULL_HalfFacet, dtype=HalfFacetType)
        Cell_HF_REF[1][0] = (2, 0)
        Cell_HF_REF[2][0] = (1, 0)
        # print(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()])
        # print(Cell_HF_REF)
        self.assertEqual(np.array_equal(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()],Cell_HF_REF), True, "Should be True.")

        # check 'Vtx2HalfFacets' against reference data
        Vtx2HalfFacets_REF = np.full(4, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        Vtx2HalfFacets_REF[0] = (0, 0, 0)
        Vtx2HalfFacets_REF[1] = (1, 1, 0)
        Vtx2HalfFacets_REF[2] = (1, 2, 0)
        Vtx2HalfFacets_REF[3] = (2, 3, 0)
        # print(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()])
        # print(Vtx2HalfFacets_REF)
        self.assertEqual(np.array_equal(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()],\
                         Vtx2HalfFacets_REF), True, "Should be True.")

        hf_in = np.array((1, 0), dtype=HalfFacetType)
        attached0 = self.Mesh.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Mesh.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached0,np.array([(1,0), (2,0)], dtype=HalfFacetType)), True, "Should be [(1,0), (2,0)].")
        hf_in[['ci','fi']] = (0, 0)
        attached1 = self.Mesh.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Mesh.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached1,np.array([(0,0)], dtype=HalfFacetType)), True, "Should be [(0,0)].")
        print("")

        attached_cl = self.Mesh.Get_Cells_Attached_To_Vertex(1)
        # print(attached_cl)
        self.assertEqual(np.array_equal(attached_cl,np.array([1, 2], dtype=CellIndType)), True, "Should be [1, 2].")
        self.assertEqual(self.Mesh.Is_Connected(1, 2), False, "Should be False.")
        self.assertEqual(self.Mesh.Is_Connected(1, 0), False, "Should be False.")

        EE = self.Mesh.Cell.Get_Edges()
        self.Mesh.Cell.Print_Edges()
        
        Edge_CHK = np.full(0, NULL_MeshEdge, dtype=MeshEdgeType)
        self.assertEqual(np.array_equal(Edge_CHK,EE), True, "Should be True.")

        cell_attached_edge0 = self.Mesh.Get_Cells_Attached_To_Edge(1, 1)
        #print(cell_attached_edge0)
        self.assertEqual(np.array_equal(cell_attached_edge0,np.array([1, 2], dtype=CellIndType)), True, "Should be [1, 2].")
        cell_attached_edge1 = self.Mesh.Get_Cells_Attached_To_Edge(0, 1)
        #print(cell_attached_edge1)
        self.assertEqual(np.array_equal(cell_attached_edge1,np.array([], dtype=CellIndType)), True, "Should be [].")

        uv = self.Mesh.Get_Unique_Vertices()
        self.Mesh.Print_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,np.array([0, 1, 2], dtype=VtxIndType)), True, "Should be [0, 1, 2].")

        free_bdy = self.Mesh.Cell.Get_FreeBoundary()
        print("free boundary of the mesh:")
        print(free_bdy)
        free_bdy_CHK = np.full(2, NULL_HalfFacet, dtype=HalfFacetType)
        free_bdy_CHK[0] = (0, 0)
        free_bdy_CHK[1] = (3, 0)
        self.assertEqual(np.array_equal(free_bdy,free_bdy_CHK), True, "Should be True.")
        print("")

        non_man_hf = self.Mesh.Get_Nonmanifold_HalfFacets()
        self.Mesh.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([], dtype=HalfFacetType)), True, "Should be [].")

        non_man_vtx = self.Mesh.Get_Nonmanifold_Vertices()
        self.Mesh.Print_Nonmanifold_Vertices()
        self.assertEqual(np.array_equal(non_man_vtx,np.array([1], dtype=VtxIndType)), True, "Should be [1].")

    def test_1D_Nonmanifold_Mesh_1(self):
        """
        We create a mesh of 5 line segments ("box with a lid"), with one
        disconnected vertex.  Embedding it into \R^2, it looks like:

                        +V4
                       /
                      C4
                     /       +V5 (and C5)
           V2+--C2--+V3
             |      |
             C3     C1
             |      |
           V0+--C0--+V1

        where the Vi are the vertex indices, and Ci are the cell indices.
        Obviously, this is not a manifold mesh.
        
        Note: there are two non-manifold "points" here: V3 and V5.  But because
        of the dimension, they are treated slightly differently.  In 1-D, half-facets
        and vertices are (topologically) the same; but in higher dimensions, this is
        not true.  Here, V3 is considered a non-manifold half-facet.  V5 is considered
        a non-manifold vertex.
        
        The PyAHF code treats non-manifold vertices as *different* from non-manifold
        half-facets in all dimensions.  The only non-manifold vertices in 1-D are
        isolated vertices.
        """
        del(self.Mesh)
        self.Mesh = BaseSimplexMesh(1)

        # see help text above
        cv = np.array([0,1, 1,3, 3,2, 2,0, 3,4, 5,5])
        cv.shape = (6,2)
        self.Mesh.Append_Cell(cv)

        self.Mesh.Finalize_Mesh_Connectivity()
        self.Mesh.Cell.Print()
        self.Mesh.Print_Vtx2HalfFacets()

        # check 'Cell.halffacet' against reference data
        Cell_HF_REF = np.full((6,2), NULL_HalfFacet, dtype=HalfFacetType)
        Cell_HF_REF[0][0] = (1, 1)
        Cell_HF_REF[0][1] = (3, 0)
        Cell_HF_REF[1][0] = (2, 1)
        Cell_HF_REF[1][1] = (0, 0)
        Cell_HF_REF[2][0] = (3, 1)
        Cell_HF_REF[2][1] = (4, 1)
        Cell_HF_REF[3][0] = (0, 1)
        Cell_HF_REF[3][1] = (2, 0)
        Cell_HF_REF[4][1] = (1, 0)
        Cell_HF_REF[5][0] = (5, 1)
        Cell_HF_REF[5][1] = (5, 0)
        #print(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()])
        #print(Cell_HF_REF)
        self.assertEqual(np.array_equal(self.Mesh.Cell.halffacet[0:self.Mesh.Num_Cell()],Cell_HF_REF), True, "Should be True.")

        # check 'Vtx2HalfFacets' against reference data
        Vtx2HalfFacets_REF = np.full(7, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        Vtx2HalfFacets_REF[0]  = (0, 0, 1)
        Vtx2HalfFacets_REF[1]  = (1, 0, 0)
        Vtx2HalfFacets_REF[2]  = (2, 2, 0)
        Vtx2HalfFacets_REF[3]  = (3, 1, 0)
        Vtx2HalfFacets_REF[4]  = (4, 4, 0)
        Vtx2HalfFacets_REF[5]  = (5, 5, 0)
        Vtx2HalfFacets_REF[6]  = (5, 5, 1)
        #print(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()])
        #print(Vtx2HalfFacets_REF)
        self.assertEqual(np.array_equal(self.Mesh.Vtx2HalfFacets.VtxMap[0:self.Mesh.Vtx2HalfFacets.Size()],\
                         Vtx2HalfFacets_REF), True, "Should be True.")

        hf_in = np.array((4, 1), dtype=HalfFacetType)
        attached0 = self.Mesh.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Mesh.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached0,np.array([(4,1), (1,0), (2,1)], dtype=HalfFacetType)), True, "Should be [(4,1), (1,0), (2,1)].")
        hf_in[['ci','fi']] = (3, 1)
        attached1 = self.Mesh.Cell.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.Mesh.Cell.Print_HalfFacets_Attached_To_HalfFacet(hf_in)
        self.assertEqual(np.array_equal(attached1,np.array([(3,1), (2,0)], dtype=HalfFacetType)), True, "Should be [(3,1), (2,0)].")
        print("")

        attached_cl = self.Mesh.Get_Cells_Attached_To_Vertex(3)
        #print(attached_cl)
        self.assertEqual(np.array_equal(attached_cl,np.array([1, 2, 4], dtype=CellIndType)), True, "Should be [1, 2, 4].")
        self.assertEqual(self.Mesh.Is_Connected(0, 2), True, "Should be True.")
        self.assertEqual(self.Mesh.Is_Connected(3, 0), False, "Should be False.")

        EE = self.Mesh.Cell.Get_Edges()
        self.Mesh.Cell.Print_Edges()
        
        Edge_CHK = np.full(6, NULL_MeshEdge, dtype=MeshEdgeType)
        Edge_CHK[0]  = (0, 1)
        Edge_CHK[1]  = (0, 2)
        Edge_CHK[2]  = (1, 3)
        Edge_CHK[3]  = (2, 3)
        Edge_CHK[4]  = (3, 4)
        Edge_CHK[5]  = (5, 5)
        self.assertEqual(np.array_equal(Edge_CHK,EE), True, "Should be True.")

        cell_attached_edge = self.Mesh.Get_Cells_Attached_To_Edge(3, 2)
        self.assertEqual(np.array_equal(cell_attached_edge,np.array([2], dtype=CellIndType)), True, "Should be [2].")

        uv = self.Mesh.Get_Unique_Vertices()
        self.Mesh.Print_Unique_Vertices()
        self.assertEqual(np.array_equal(uv,np.array([0, 1, 2, 3, 4, 5], dtype=VtxIndType)), True, "Should be [0, 1, 2, 3, 4, 5].")

        free_bdy = self.Mesh.Cell.Get_FreeBoundary()
        print("free boundary of the mesh:")
        print(free_bdy)
        free_bdy_CHK = np.full(1, NULL_HalfFacet, dtype=HalfFacetType)
        free_bdy_CHK[0] = (4, 0)
        self.assertEqual(np.array_equal(free_bdy,free_bdy_CHK), True, "Should be True.")
        print("")

        non_man_hf = self.Mesh.Get_Nonmanifold_HalfFacets()
        self.Mesh.Print_Nonmanifold_HalfFacets()
        self.assertEqual(np.array_equal(non_man_hf,np.array([(4, 1)], dtype=HalfFacetType)), True, "Should be [(4, 1)].")

        non_man_vtx = self.Mesh.Get_Nonmanifold_Vertices()
        self.Mesh.Print_Nonmanifold_Vertices()
        self.assertEqual(np.array_equal(non_man_vtx,np.array([5], dtype=VtxIndType)), True, "Should be [5].")


    # def test_2D_Nonmanifold_Mesh_1(self):
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

    # def test_2D_Nonmanifold_Mesh_2(self):
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

if __name__ == '__main__':
    unittest.main()
