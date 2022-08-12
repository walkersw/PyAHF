"""
ahf.BasicClasses.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Definition of small light-weight classes and data types used to hold
mesh cell data, mesh vertex coordinate data, etc.

Note: these structs are used within BaseSimplexMesh and ???, for storing
      cell and point coordinate data.

Copyright (c) 08-11-2022,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, PointType
from ahf import *

from ahf.Vtx2HalfFacet_Mapping import *

# define data types

# a single mesh edge (two connected vertices)
# An edge is defined by [v0, v1], where v0, v1 are the end vertices of the edge.
# For a simplex mesh, the edge [v0, v1] exists when v0, v1 are both
# contained in the *same* mesh cell.
MeshEdgeType = np.dtype({'names': ['v0', 'v1'],
                         'formats': [VtxIndType, VtxIndType],
                         'titles': ['global tail index', 'global head index']})
NULL_MeshEdge = np.array((NULL_Vtx, NULL_Vtx), dtype=MeshEdgeType)


class CellSimplexType:
    """
    Class for storing cell (simplex) connectivity data and sibling half-facets.
    CELL_DIM is the topological dimension of the cell (which is assumed to be a simplex).
    CELL_DIM = 0: cell is a single vertex (point);
                  half-facets are NULL (or can view as the point itself).
    CELL_DIM = 1: cell is an edge (line segment);
                  half-facets are end-point vertices of the edge.
    CELL_DIM = 2: cell is a triangle;
                  half-facets are edges of the triangle.
    CELL_DIM = 3: cell is a tetrahedron;
                  half-facets are faces of the tetrahedron.
    One can continue increasing the dimension...

    REMARK: We need to define a one-to-one mapping between *local* facets and
    *local* vertices of a cell.  Also, see "BaseSimplexMesh.cc" for the definition of
    what a local vertex and local facet are.
   
    Note: Vi (vtx) is *never* contained in Fi (facet), because Fi is always opposite Vi.

    For a point (CELL_DIM==0), we define this to be:
      Vtx | Facet (Vertex)
     -----+-------------
       0  | NULL (or 0)  (Note: this is either nothing or Vtx #0)

    For an edge (CELL_DIM==1), we define this to be:
      Vtx | Facet (Vertex)
     -----+-------------
       0  |   1   (Note: this is actually Vtx #0)
       1  |   0   (Note: this is actually Vtx #1)

    For a triangle (CELL_DIM==2), we define this to be:
      Vtx | Facet (Edge)
     -----+-------------
       0  |   1
       1  |   2
       2  |   0

    For a tetrahedron (CELL_DIM==3), we define this to be:
      Vtx | Facet (Face)
     -----+-------------
       0  |   1
       1  |   2
       2  |   3
       3  |   0

    (The pattern is obvious for higher cell dimensions...)

    Note: each vertex is contained (attached) to its corresponding facet.
    """

    def __init__(self,CELL_DIM):

        self._cell_dim = CELL_DIM
        # global vertex indices of the cells (initialize)
        self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        # half-facets corresponding to local facets of the cells (initialize)
        self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        # actual number of cells
        self._size = 0

    def __str__(self):
        dimvtx = self.vtx.shape
        dimhf  = self.halffacet.shape
        if not np.array_equal(dimvtx,dimhf):
            print("vtx and halffacet arrays are not the same size!")
        OUT_STR = ("The number of cells is: " + str(self._size) + "\n"
                 + "The *reserved* size of cells is: " + str(dimvtx[0]) + "\n" )
        return OUT_STR

    def Clear(self):
        del(self.vtx)
        del(self.halffacet)
        self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        self._size = 0

    def Size(self):
        return self._size

    def Cell_Dim(self):
        return self._cell_dim

    def Reserve(self, num_cl):
        """This just pre-allocates, or re-sizes.
         The _size attribute is unchanged."""
        dimvtx = self.vtx.shape
        dimhf  = self.halffacet.shape
        if not np.array_equal(dimvtx,dimhf):
            print("vtx and halffacet arrays are not the same size!")

        # compute the space needed
        Desired_Size = np.rint(num_cl).astype(VtxIndType)
        
        if dimvtx[0] < Desired_Size:
            old_size = dimvtx[0]
            self.vtx.resize((Desired_Size, self._cell_dim+1))
            self.halffacet.resize((Desired_Size, self._cell_dim+1))
            # put in NULL values
            self.vtx[old_size:Desired_Size][:]       = NULL_Vtx
            self.halffacet[old_size:Desired_Size][:] = NULL_HalfFacet
        else:
            pass

    def Append_Cell(self, vtx_ind):
        """Append a single cell by giving its global vertex indices (as an array).
        """
        
        dimvtx = self.vtx.shape
        if (dimvtx[0]==self._size):
            # need to reserve space
            Reserve(self, self._size+10)
        
        if len(vtx_ind)==(self._cell_dim+1):
            self.vtx[self._size][:] = vtx_ind[:]
            self._size += 1
        else:
            print("Error: incorrect number of vertex indices!")

    def Append_Cell_Data(self, num_cells, vtx_ind):
        """Append several cells at once by giving their global vertex
        indices (as an array).
        """
        
        Num_Current_Cells = self.Size()
        New_Total_Cells = Num_Current_Cells + num_cells;
        self.Reserve(New_Total_Cells)
        
        if len(vtx_ind)==(self._cell_dim+1)*num_cells:
            dimvtx = self.vtx.shape
            self.vtx.shape = (dimvtx[0]*dimvtx[1],)
            self.vtx[Num_Current_Cells*(self._cell_dim+1):New_Total_Cells*(self._cell_dim+1)] = vtx_ind[:]
            self._size += num_cells
            # put it back
            self.vtx.shape = dimvtx
        else:
            print("Error: incorrect number of vertex indices!")

    def Set_Cell(self, cell_ind, vtx_ind):
        """Set the vertex data for a given cell (that already exists) by giving its
        global vertex indices (as an array).
        """
        
        if (cell_ind>=self._size):
            print("Error: the given cell index does not exist!")
        
        if len(vtx_ind)==(self._cell_dim+1):
            self.vtx[cell_ind][:] = vtx_ind[:]
        else:
            print("Error: incorrect number of vertex indices!")

    def Set_Cell_Data(self, num_cells, vtx_ind):
        """Set all cell data at once.
        """
        
        self.Reserve(num_cells)
        
        if len(vtx_ind)==(self._cell_dim+1)*num_cells:
            dimvtx = self.vtx.shape
            self.vtx.shape = (dimvtx[0]*dimvtx[1],)
            self.vtx[0:num_cells*(self._cell_dim+1)] = vtx_ind[:]
            self._size = num_cells
            # put it back
            self.vtx.shape = dimvtx
        else:
            print("Error: incorrect number of vertex indices!")


    # def Get_Unique_Vertices(self):
        # """Get unique list of vertices.
        # Does not require 'Sort()' to have been run.
        # """
        
        # all_vertices = np.zeros(self._size, dtype=VtxIndType)
        # for kk in range(self._size):
            # # extract out the vertex indices
            # all_vertices[kk] = self.VtxMap[kk]['vtx']

        # unique_vertices = np.unique(all_vertices)
        # return unique_vertices

    # def Display_Unique_Vertices(self):
        # """Print unique list of vertices to the screen.
        # Does not require 'Sort()' to have been run.
        # """
        
        # unique_vertices = self.Get_Unique_Vertices()
        
        # print("Unique list of vertex indices:")
        # print(str(unique_vertices[0]), end="")
        # for kk in range(1, unique_vertices.size):
            # print(", " + str(unique_vertices[kk]), end="")
        # print("")

    # def Sort(self):
        # """Sort the VtxMap so it is useable."""
        # self.VtxMap = np.sort(self.VtxMap, order=['vtx', 'ci', 'fi'])

