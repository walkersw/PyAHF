"""
ahf.BasicClasses.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Definition of small, light-weight classes and data types used to hold
mesh cell data, mesh vertex coordinate data, etc.

Note: these structs are used within BaseSimplexMesh and ???, for storing
      cell and point coordinate data.

Copyright (c) 08-12-2022,  Shawn W. Walker
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

        if (CELL_DIM<0):
            print("Error: cell dimension must be non-negative!")
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
        OUT_STR = ("The topological dimension is: " + str(self._cell_dim) + "\n"
                 + "The number of cells is: " + str(self._size) + "\n"
                 + "The *reserved* size of cells is: " + str(dimvtx[0]) + "\n")
        return OUT_STR

    def Clear(self):
        """This clears all cell data.
         The _size attribute (i.e. number of cells) is set to zero."""
        del(self.vtx)
        del(self.halffacet)
        self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        self._size = 0

    def Size(self):
        """This returns the number of cells."""
        return self._size

    def Dim(self):
        """This returns the topological dimension of the cells."""
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

    def Append(self, vtx_ind):
        """Append a single cell by giving its global vertex indices (as an array).
        """
        
        dimvtx = self.vtx.shape
        if (dimvtx[0]==self._size):
            # need to reserve space
            self.Reserve(self._size+10)
        
        if len(vtx_ind)==(self._cell_dim+1):
            self.vtx[self._size][:] = vtx_ind[:]
            self._size += 1
        else:
            print("Error: incorrect number of vertex indices!")

    def Append_Batch(self, num_cells, vtx_ind):
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

    def Set(self, cell_ind, vtx_ind):
        """Set the vertex data for a given cell (that already exists) by giving its
        global vertex indices (as an array).
        """
        
        if (cell_ind>=self._size):
            print("Error: the given cell index does not exist!")
        
        if len(vtx_ind)==(self._cell_dim+1):
            self.vtx[cell_ind][:] = vtx_ind[:]
        else:
            print("Error: incorrect number of vertex indices!")

    def Set_All(self, num_cells, vtx_ind):
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

    def _get_cell_string(self, ci):
        """Get string representing single cell data.
        """
        if (ci >= self._size):
            print("Error: invalid cell index!")
            OUT = "INVALID"
        else:
            vtx = self.vtx[ci][:]
            hfs = self.halffacet[ci][:]
            OUT = "[" + str(vtx[0])
            for kk in range(1, (self._cell_dim+1)):
                OUT += ", " + str(vtx[kk])
            OUT += "] | ["
            if (hfs[0]['ci']==NULL_Cell):
                hf0_str = "(NULL)"
            else:
                hf0_str = "(" + str(hfs[0]['ci']) + ", " + str(hfs[0]['fi']) + ")"
            OUT += hf0_str
            for kk in range(1, (self._cell_dim+1)):
                if (hfs[kk]['ci']==NULL_Cell):
                    hfkk_str = "(NULL)"
                else:
                    hfkk_str = "(" + str(hfs[kk]['ci']) + ", " + str(hfs[kk]['fi']) + ")"
                OUT += ", " + hfkk_str
            OUT += "]"
        return OUT

    def Print_Cell(self, ci):
        """Print single cell data to the screen.
        """
        if (ci >= self._size):
            print("Error: invalid cell index!")
        else:
            OUT = self._get_cell_string(ci)
            print(OUT)

    def Print(self):
        """Print all the cell data to the screen.
        """
        
        print("Cell connectivity data:")
        print("-----------------------")
        print("cell index: [vertex indices] | [sibling half-facets]")
        for ci in range(0, self._size):
            cell_str = self._get_cell_string(ci)
            OUT_str = str(ci) + ": " + cell_str
            print(OUT_str)
        print("")

    def Get_Unique_Vertices(self):
        """Get unique list of vertices from the current cell data.
        """
        
        sub_cell = self.vtx[0:self._size][:]
        dimsub = sub_cell.shape
        sub_cell.shape = (dimsub[0]*dimsub[1],)
        unique_vertices = np.unique(sub_cell)
        
        return unique_vertices

    def Print_Unique_Vertices(self):
        """Print unique list of vertices (from cell data) to the screen.
        """
        
        unique_vertices = self.Get_Unique_Vertices()
        
        print("Unique list of vertex indices:")
        print(str(unique_vertices[0]), end="")
        for kk in range(1, unique_vertices.size):
            print(", " + str(unique_vertices[kk]), end="")
        print("")

    # def Sort(self):
        # """Sort the VtxMap so it is useable."""
        # self.VtxMap = np.sort(self.VtxMap, order=['vtx', 'ci', 'fi'])


class VtxCoordType:
    """
    Class for storing vertex coordinate data.
    GEO_DIM >= 1 is the geometric (ambient) dimension that the points live in.
    """

    def __init__(self,GEO_DIM):

        if (GEO_DIM<1):
            print("Error: geometric dimension must be positive!")
        self._geo_dim = GEO_DIM
        # vertex coordinates (initialize)
        self.coord = np.full((1, self._geo_dim), 0.0, dtype=PointType)
        # actual number of vertices
        self._size = 0

    def __str__(self):
        dimcoord = self.coord.shape
        OUT_STR = ("The geometric dimension is: " + str(self._geo_dim) + "\n"
                 + "The number of vertices is: " + str(self._size) + "\n"
                 + "The *reserved* size of vertices is: " + str(dimcoord[0]) + "\n")
        return OUT_STR

    def Clear(self):
        """This clears all coordinate data.
         The _size attribute (i.e. number of vertices) is set to zero."""
        del(self.coord)
        self.coord = np.full((1, self._geo_dim), 0.0, dtype=PointType)
        self._size = 0

    def Size(self):
        """This returns the number of vertices."""
        return self._size

    def Dim(self):
        """This returns the geometric dimension of the points."""
        return self._geo_dim

    def Reserve(self, num_vtx):
        """This just pre-allocates, or re-sizes.
         The _size attribute is unchanged."""
        dimcoord = self.coord.shape

        # compute the space needed
        Desired_Size = np.rint(num_vtx).astype(VtxIndType)
        
        if dimcoord[0] < Desired_Size:
            old_size = dimcoord[0]
            self.coord.resize((Desired_Size, self._geo_dim))
            # put in ZERO values
            self.coord[old_size:Desired_Size][:] = 0.0
        else:
            pass

    def Append(self, vtx_coord):
        """Append a single vertex by giving its coordinates (as an array).
        """
        
        dimcoord = self.coord.shape
        if (dimcoord[0]==self._size):
            # need to reserve space
            self.Reserve(self._size+10)
        
        if len(vtx_coord)==(self._geo_dim):
            self.coord[self._size][:] = vtx_coord[:]
            self._size += 1
        else:
            print("Error: incorrect number of vertex coordinates!")

    def Append_Batch(self, num_vtx, vtx_coord):
        """Append several vertices at once by giving their coordinates (as an array).
        """
        
        Num_Current_Vtx = self.Size()
        New_Total_Vtx = Num_Current_Vtx + num_vtx;
        self.Reserve(New_Total_Vtx)
        
        if len(vtx_coord)==self._geo_dim*num_vtx:
            dimcoord = self.coord.shape
            self.coord.shape = (dimcoord[0]*dimcoord[1],)
            self.coord[Num_Current_Vtx*self._geo_dim:New_Total_Vtx*self._geo_dim] = vtx_coord[:]
            self._size += num_vtx
            # put it back
            self.coord.shape = dimcoord
        else:
            print("Error: incorrect number of vertex coordinates!")

    def Set(self, vtx_ind, vtx_coord):
        """Set the coordinates for a given vertex (that already exists) by giving an array.
        """
        
        if (vtx_ind>=self._size):
            print("Error: the given vertex index does not exist!")
        
        if len(vtx_coord)==self._geo_dim:
            self.coord[vtx_ind][:] = vtx_coord[:]
        else:
            print("Error: incorrect number of vertex coordinates!")

    def Set_All(self, num_vtx, vtx_coord):
        """Set all vertex coordinate at once.
        """
        
        self.Reserve(num_vtx)
        
        if len(vtx_coord)==self._geo_dim*num_vtx:
            dimcoord = self.coord.shape
            self.coord.shape = (dimcoord[0]*dimcoord[1],)
            self.coord[0:num_vtx*self._geo_dim] = vtx_coord[:]
            self._size = num_vtx
            # put it back
            self.coord.shape = dimcoord
        else:
            print("Error: incorrect number of vertex coordinates!")

    def _get_coord_string(self, vi, num_digit=3):
        """Get string representing single vertex coordinate data.
        Optional extra argument specifies number of digits after decimal point.
        """
        if (vi >= self._size):
            print("Error: invalid vertex index!")
            OUT = "INVALID"
        else:
            F_str = "{:." + str(num_digit) + "f}"
            
            coord = self.coord[vi][:]
            OUT = "[" + F_str.format(coord[0])
            for kk in range(1, self._geo_dim):
                OUT += ", " + F_str.format(coord[kk])
            OUT += "]"
        return OUT

    def Print_Vtx(self, vi, num_digit=3):
        """Print single vertex coordinate data to the screen.
        Optional extra argument specifies number of digits after decimal point.
        """
        if (vi >= self._size):
            print("Error: invalid vertex index!")
        else:
            OUT = self._get_coord_string(vi, num_digit)
            print(OUT)

    def Print(self, num_digit=3):
        """Print all the vertex coordinate data to the screen.
        Optional argument specifies number of digits after decimal point.
        """
        
        print("Vertex coordinate data:")
        print("-----------------------")
        print("vtx index: [x_{0}, ..., x_{" + str(self._geo_dim-1) + "}]")
        for vi in range(0, self._size):
            vtx_str = self._get_coord_string(vi, num_digit)
            OUT_str = str(vi) + ": " + vtx_str
            print(OUT_str)
        print("")


    #