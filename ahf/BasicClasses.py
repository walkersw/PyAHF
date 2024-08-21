"""
ahf.BasicClasses.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Definition of small, light-weight classes and data types used to hold
mesh cell data, mesh vertex coordinate data, etc.

Note: these structs are used within BaseSimplexMesh and SimplexMesh, for
      storing cell and point coordinate data.

Copyright (c) 07-24-2024,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
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

    def __init__(self, CELL_DIM, res_buf=0.2):
        """
        CELL_DIM >= 0 is the topological dimension that the cells live in.
        res_buf in [0.0, 1.0] is optional, and is for reserving extra space for cell data.
        """
        if (CELL_DIM<0):
            print("Error: cell dimension must be non-negative!")
        assert(CELL_DIM>=0)
        if np.rint(CELL_DIM).astype(SmallIndType)!=CELL_DIM:
            print("Error: cell dimension must be a non-negative integer!")
        assert(np.rint(CELL_DIM).astype(SmallIndType)==CELL_DIM)

        self._cell_dim = CELL_DIM
        # global vertex indices of the cells (initialize)
        self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        # half-facets corresponding to local facets of the cells (initialize)
        self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        # actual number of cells
        self._size = 0
        # amount of extra memory to allocate when re-allocating vtx and halffacet
        #       (number between 0.0 and 1.0).
        self._reserve_buffer = res_buf # e.g. 0.2 means extra 20%

    def __str__(self):
        dimvtx = self.vtx.shape
        dimhf  = self.halffacet.shape
        if not np.array_equal(dimvtx,dimhf):
            print("vtx and halffacet arrays are not the same size!")
        OUT_STR = ("The topological dimension is: " + str(self._cell_dim) + "\n"
                 + "The number of cells is: " + str(self._size) + "\n"
                 + "The *reserved* size of cells is: " + str(self.Capacity()) + "\n")
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

        # compute the actual size to allocate for the cells
        Desired_Size = np.rint(np.ceil((1.0 + self._reserve_buffer) * num_cl)).astype(CellIndType)
        if dimvtx[0] < Desired_Size:
            old_size = dimvtx[0]
            self.vtx.resize((Desired_Size, self._cell_dim+1))
            self.halffacet.resize((Desired_Size, self._cell_dim+1))
            # put in NULL values
            self.vtx[old_size:Desired_Size][:]       = NULL_Vtx
            self.halffacet[old_size:Desired_Size][:] = NULL_HalfFacet
        else:
            pass

    def Capacity(self):
        """This just returns the reserved number of cells."""
        dimvtx = self.vtx.shape
        dimhf  = self.halffacet.shape
        if not np.array_equal(dimvtx,dimhf):
            print("Error: vtx and halffacet arrays are not the same size!")
        return dimvtx[0]

    def Append(self, cell_vtx):
        """Append several cells at once by giving their global vertex indices (as a numpy array).
        cell_vtx has shape (M,self.Dim()+1), where M is the number of cells, or
        cell_vtx is a 1-D array of length self.Dim()+1 in the case of one cell.
        """
        if type(cell_vtx) is not np.ndarray:
            print("Error: input must be a numpy array!")
            return

        dim_cl = cell_vtx.shape
        if len(dim_cl)==1:
            num_cell = 1
            input_cell_dim = dim_cl[0] - 1
        else:
            num_cell = dim_cl[0]
            input_cell_dim = dim_cl[1] - 1

        Num_Current_Cells = self.Size()
        New_Total_Cells = Num_Current_Cells + num_cell
        self.Reserve(New_Total_Cells)
        
        if input_cell_dim==self._cell_dim:
            self.vtx[Num_Current_Cells:New_Total_Cells][:] = cell_vtx
            self._size += num_cell
        else:
            print("Error: incorrect number of vertex indices!")

    def Set(self, *args):
        """Set cell data.  Two ways to call:
        -Set all cell data at once:
            one input: cell_vtx, which has shape (M,self.Dim()+1), where M is the number of cells.
        -Overwrite cell vertex indices of specific cell:
            two inputs: (cell_ind, cell_vtx), where
            cell_ind is a single cell index that already exists,
            cell_vtx is a numpy array of length self.Dim()+1
        """
        # decipher inputs
        if len(args)==1:
            # set all vertex coordinates
            cell_vtx = args[0]
            if type(cell_vtx) is not np.ndarray:
                print("Error: cell_vtx input must be a numpy array!")
                return
            dim_cl = cell_vtx.shape
            if len(dim_cl)==1:
                num_cell = 1
                input_cell_dim = dim_cl[0] - 1
            else:
                num_cell = dim_cl[0]
                input_cell_dim = dim_cl[1] - 1
        elif len(args)==2:
            # set a specific cell
            cell_ind = args[0]
            cell_vtx = args[1]
            if type(cell_vtx) is not np.ndarray:
                print("Error: cell_vtx input must be a numpy array!")
                return

            input_cell_dim = cell_vtx.size - 1
            if ( (cell_ind==NULL_Cell) or (cell_ind>=self._size) ):
                print("Error: given cell index is not valid!")
                return
        else:
            print("incorrect number of arguments!")
            return

        if input_cell_dim==self._cell_dim:
            if len(args)==1:
                self.Reserve(num_cell)
                self.vtx[0:num_cell,:] = cell_vtx
                self._size = num_cell
            else:
                self.vtx[cell_ind,:] = cell_vtx
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

    def Print(self, ci=NULL_Cell):
        """Print cell connectivity and sibling half-facets. "ci" is the index of a
        specific cell; if ci=NULL_Cell, then print all cells.  If no cell index is
        given, then print all cells.
        """
        if (ci==NULL_Cell):
            # then print all the cells
            print("Cell connectivity data:")
            print("-----------------------")
            print("cell #:        vertices        |   sibling half-facets")
            NC = self.Size()
            for kk in np.arange(0, NC, dtype=CellIndType):
                cell_str = self._get_cell_string(kk)
                OUT_str = str(kk) + ": " + cell_str
                print(OUT_str)
        else:
            # then print ONE cell
            if (ci >= self._size):
                print("Error: invalid cell index!")
                return
            print("Connectivity of cell #: " + str(ci))
            print("----------------------------------")
            print("        vertices        |   sibling half-facets")
            cell_str = self._get_cell_string(ci)
            print(cell_str)
        print("")

    def Get_Unique_Vertices(self):
        """Get unique list of vertices from the current cell data.
        """
        unique_vertices = np.unique(self.vtx[0:self.Size()][:])
        
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

    def Num_Vtx(self):
        """Returns the number of (unique) vertices referenced in self.vtx."""
        uv = self.Get_Unique_Vertices()
        return uv.size

    def Max_Vtx_Index(self):
        """Returns the largest vertex index referenced in self.vtx."""
        return np.amax(self.vtx[0:self.Size()][:])
        
    def Reindex_Vertices(self, new_indices):
        """Re-index the vertices in the mesh.
        Example: new_index = new_indices[old_index]
        """
        # basic check
        if (new_indices.size < self.Max_Vtx_Index()):
            print("Error in 'CellSimplexType.Reindex_Vertices'!")
            print("    The given list of indices is shorter than the")
            print("    max vertex index referenced by cells in the mesh.")
            return

        # go through all the cells, and the vertices of each cell, and map those vertices
        NC = self.Size()
        for ci in np.arange(0, NC, dtype=CellIndType):
            for vi in np.arange(0, self._cell_dim+1, dtype=SmallIndType):
                self.vtx[ci][vi] = new_indices[self.vtx[ci][vi]]

    def Get_Edges(self):
        """Returns numpy array, of length M and type MeshEdgeType, containing all edges
        in the mesh, where M is the number of edges.  Note: this is a unique list of
        *sorted* edges, i.e. each element is an edge (v0, v1) that satisfies v0 < v1,
        where v0, v1 are the global vertex indices of the edge end points.
        Moreover, the edges are in ascending order (['v0', 'v1']).
        """
        NC = self.Size()
        Num_Local_Edge = np.ceil((self._cell_dim+1) * self._cell_dim / 2)
        S = NC * Num_Local_Edge
        #edges = np.zeros((S,2), dtype=VtxIndType)
        
        edges = np.full(S.astype(CellIndType), NULL_MeshEdge, dtype=MeshEdgeType)
        
        # add all edges of each cell
        ind_cnt = np.zeros(1, dtype=CellIndType)
        for ci in np.arange(0, NC, dtype=CellIndType):
            # loop through each local edge of the current (simplex) cell
            # i.e. loop through all the distinct pairs of vertices in the cell
            for vi in np.arange(0, self._cell_dim+1, dtype=SmallIndType):
                c_vtx = np.sort(self.vtx[ci]) # sort whole cell
                for vj in np.arange(vi+1, self._cell_dim+1, dtype=SmallIndType):
                    edges[ind_cnt] = (c_vtx[vi], c_vtx[vj])
                    ind_cnt += 1

        # check
        if ind_cnt!=S:
            print("Error: we did not get all the edges!?!?")

        # sort all the edges
        edges = np.sort(edges, order=['v0', 'v1'])
        edges = np.unique(edges)

        return edges

    def Print_Edges(self):
        """Print out all the edges in the mesh (this uses "Get_Edges").
        """
        edges = self.Get_Edges()
        
        if (edges.size > 0):
            # print all mesh edges
            print("All edges of the mesh (a unique, sorted list):")
            print("Edge #:  ( vertex #0, vertex #1 )")
            for ei in np.arange(0, edges.size, dtype=CellIndType):
                EE_str = self._get_edge_string(edges[ei]['v0'], edges[ei]['v1'])
                OUT_str = str(ei) + ": " + EE_str
                print(OUT_str)
        else:
            print("There are NO mesh edges!")
        
        print(" ")

    def _get_edge_string(self, v0, v1):
        """Get string representing an edge.
        """
        if ( (v0==NULL_Vtx) or (v1==NULL_Vtx)):
            EE_str = "(NULL)"
        else:
            EE_str = "(" + str(v0) + ", " + str(v1) + ")"
        return EE_str

    def Get_FreeBoundary(self):
        """Returns all half-facets that are referenced by only one cell;
        i.e. the half-facets that are on the boundary of the mesh.
        WARNING: this requires the sibling half-facet data (see self.halffacet)
        to be built before this can be used correctly.
        Note: the returned numpy array can be empty.
        """
        NC = self.Size()
        bdy = [] # init to empty list

        # check all cells
        for ci in np.arange(0, NC, dtype=CellIndType):
            Cell_HFs = self.halffacet[ci]
            # loop through each local facet of the current (simplex) cell
            for fi in np.arange(0, self._cell_dim + 1, dtype=SmallIndType):
                HF = np.array((NULL_Cell, NULL_Small), dtype=HalfFacetType)
                if (Cell_HFs[fi]==NULL_HalfFacet):
                    # this facet has no neighbor!
                    HF[['ci','fi']] = (ci, fi) # so store this bdy facet
                    bdy.append(HF)

        bdy_np = np.array(bdy, dtype=HalfFacetType)
        return bdy_np

    def Get_Cells_Attached_To_Vertex(self, vi, ci):
        """Returns all cell indices (a numpy array) that are attached to vertex "vi" AND are
        facet-connected to the cell "ci".  WARNING: this requires the sibling half-facet data
        (see self.halffacet) to be built before this can be used.
        Note: the returned cell_array can be empty.
        """
        if ( (vi==NULL_Vtx) or (ci==NULL_Cell) ):
            # the vertex or starting cell is null, so do nothing.
            return np.array([], dtype=CellIndType)

        # do a recursive search to collect all the cells
        cell_list = [] # initialize as an empty list
        self._Get_Cells_Attached_To_Vertex_Recurse(vi, ci, cell_list)
        
        return np.array(cell_list, dtype=CellIndType)

    def _Get_Cells_Attached_To_Vertex_Recurse(self, vi, ci, cell_list):
        """Recursive function call for the above method.
        """
        # if we have been searching too long...
        if (len(cell_list) > 100000):
            # then quit!
            print("Error in 'Get_Cells_Attached_To_Vertex'...")
            print("    Recursion depth is too great.")
            print("    There should not be more than 100,000 cells attached to a single vertex!")
            return

        # cell is null, so nothing left to do
        if (ci==NULL_Cell):
            # this can happen if the neighbor cell does not exist
            return

        # if ci is already in the list
        if (ci in cell_list):
            # then we have already visited this cell, so done
            return
        else:
            # access the cell
            cell_vtx       = self.vtx[ci]
            cell_halffacet = self.halffacet[ci]
            # check again that the vertex is actually in the cell
            local_vi = self.Get_Local_Vertex_Index_In_Cell(vi, cell_vtx)
            if (local_vi==NULL_Small):
                # cell does not actually contain the vertex
                #    will only happen if the (vi, ci) given to 'Get_Cells_Attached_To_Vertex'
                #    is not valid, i.e. ci does not contain vi.
                # so, we are done
                return

            # add the cell to the list
            cell_list.append(ci)

            if (self._cell_dim > 0):
                # get the local facets that share the vertex
                local_facet = self.Get_Local_Facets_Sharing_Local_Vertex(local_vi)
                
                # loop through the facets
                for fi in range(self._cell_dim):
                    if (local_facet[fi]!=NULL_Small):
                        # determine the neighbor cell on the "other side" of that facet and search it
                        self._Get_Cells_Attached_To_Vertex_Recurse(vi, cell_halffacet[local_facet[fi]]['ci'], cell_list)
                    # else the neighbor does not exist, so do nothing

    def Two_Cells_Are_Facet_Connected(self, vi, ci_a, ci_b):
        """Returns true/false if two cells (that share the same vertex) are connected
        by a "chain" of half-facet neighbors.  WARNING: this requires the sibling
        half-facet data (see self.halffacet) to be built before this can be used.
        Note: this routine is useful for determining when two cells are in the same
        "connected component" of the mesh (this is important when the mesh is
        *not* a manifold).
        """
        if ( (vi==NULL_Vtx) or (ci_a==NULL_Cell) or (ci_b==NULL_Cell) ):
            # something is null, so do nothing
            return False

        # init recursion depth count
        Depth_Count = 0
        # init current cell and target cell
        Start_Cell   = np.array(ci_a, dtype=CellIndType, copy=True)
        Current_Cell = np.array(ci_a, dtype=CellIndType, copy=True)
        Target_Cell  = np.array(ci_b, dtype=CellIndType, copy=True)

        # do a recursive search to find the target
        return self._Two_Cells_Are_Facet_Connected_Recurse(vi, Start_Cell, Current_Cell, Target_Cell, Depth_Count)

    def _Two_Cells_Are_Facet_Connected_Recurse(self, vi, start, current, target, Depth_Count):
        """Recursive function call for the above method.
        """
        # if we have been searching too long...
        if (Depth_Count > 100000):
            # then quit!
            print("Error in 'Two_Cells_Are_Facet_Connected'...")
            print("    Recursion depth is too great.")
            print("    There should not be more than 100,000 cells attached to a single vertex!")
            return False

        # if we loop back to the beginning...
        if (Depth_Count > 0):
            if (current==start):
                # we did not find the target
                return False

        # current cell is null, so nothing left to do
        if (current==NULL_Cell):
            # this can happen if the neighbor cell does not exist
            return False

        # if current matches the target
        if (current==target):
            # we found it!
            return True
        else:
            # access the cell
            cell_vtx       = self.vtx[current]
            cell_halffacet = self.halffacet[current]
            # check again that the vertex is actually in the cell
            local_vi = self.Get_Local_Vertex_Index_In_Cell(vi, cell_vtx)
            if (local_vi==NULL_Small):
                # cell does not actually contain the vertex
                #    will only happen if the (vi, ci_a), or (vi, ci_b) given to
                #    'Two_Cells_Are_Facet_Connected' is not valid,
                #    i.e. ci_a does not contain vi, or ci_b does not contain vi
                # so, we are done
                return False

            # get the local facets that share the vertex
            local_facet = self.Get_Local_Facets_Sharing_Local_Vertex(local_vi)

            # keep counting
            Depth_Count += 1

            # loop through the facets
            CONNECTED = False # init
            for fi in range(self._cell_dim):
                if (local_facet[fi]!=NULL_Small):
                    # determine the neighbor cell on the "other side" of that facet and search it
                    CONNECTED = self._Two_Cells_Are_Facet_Connected_Recurse( \
                                      vi, start, cell_halffacet[local_facet[fi]]['ci'], target, Depth_Count)
                    if CONNECTED:
                        break # it was found!
                # else the neighbor does not exist, so do nothing

            return CONNECTED

    def Print_Two_Cells_Are_Facet_Connected(self, vi, ci_a, ci_b):
        """Print information stating whether two cells (that share the same vertex) are
        facet-connected. Note: this is useful for determining when two cells are in the
        same "connected component" of the mesh (this is important when the mesh
        is *not* a manifold).
        """
        if (vi==NULL_Vtx):
            # something is null, so do nothing
            print("Vertex is invalid, so print nothing!")
            return
        if ( (ci_a==NULL_Cell) or (ci_b==NULL_Cell) ):
            # cell is null, so do nothing.
            print("One of the cells is invalid, so print nothing!")
            return

        CONNECTED = self.Two_Cells_Are_Facet_Connected(vi, ci_a, ci_b)
        if (CONNECTED):
            print("Cell #" + str(ci_a) + " and Cell #" + str(ci_b) \
                  + " are facet-connected *and* share vertex #" + str(vi) + ".")
        else:
            # make sure both cells contain the vertex
            CL_a_vtx = self.vtx[ci_a]
            contain_a = np.isin(vi, CL_a_vtx)
            CL_b_vtx = self.vtx[ci_b]
            contain_b = np.isin(vi, CL_b_vtx)

            if ( contain_a and contain_b ):
                print("Cell #" + str(ci_a) + " and Cell #" + str(ci_b) + " both share vertex #" \
                      + str(vi) + " but are *not* facet connected.")
            else:
                print("Cell #" + str(ci_a) + " and Cell #" + str(ci_b) \
                      + " do *not* both share vertex #" + str(vi) + ".")

    def Get_HalfFacets_Attached_To_HalfFacet(self, hf_in):
        """Get all half-facets attached to a given half-facet.  Note that all of these
        attached half-facets refer to the *same* geometrically defined facet in the mesh.

        The output of this method is a numpy array.
        Note: the output also contains the given half-facet.
        Note: this routine requires the sibling half-facet data to be built first.
        """
        if hf_in.dtype!=HalfFacetType:
            print("Error: given half-facet must be of type HalfFacetType!")
            return []
        
        # verify that the given half-facet is not NULL
        if (hf_in==NULL_HalfFacet):
            return np.full(0, NULL_HalfFacet, dtype=HalfFacetType)

        # put in the initial half-facet
        attached_hf = [] # init to empty list
        attached_hf.append(hf_in)

        # cycle through all the neighbors and store them
        COUNT = 0
        while (COUNT < 100000): # allow for up to 100,000 neighbors!
            COUNT += 1
            # get the next half-facet
            current_hf = attached_hf[-1]
            CL_halffacet = self.halffacet[current_hf['ci']]
            next_hf = CL_halffacet[current_hf['fi']]

            # if the neighbor does not exist, then stop!
            if (next_hf==NULL_HalfFacet):
                break

            # if we get back to the starting half-facet, stop!
            if (next_hf==hf_in):
                break
            # else store it
            attached_hf.append(next_hf)
            
        if (COUNT >= 100000):
            # then quit!
            print("Error in 'CellSimplexType.Get_HalfFacets_Attached_To_HalfFacet'...")
            print("    Number of neighbors is too large.")
            print("    There should not be more than 100,000 cells attached to a single facet!")

        return np.array(attached_hf, dtype=HalfFacetType)

    def Print_HalfFacets_Attached_To_HalfFacet(self, hf_in):
        """Print half-facets attached to given half-facet.
        """
        attached = self.Get_HalfFacets_Attached_To_HalfFacet(hf_in)
        
        print("The half-facets attached to " + str(hf_in) + " are:")
        for hf_jj in attached:
            print(str(hf_jj))

    def Get_Nonmanifold_HalfFacets(self):
        """Get a unique set of non-manifold half-facets. This returns a numpy array
        of half-facets, each defining a *distinct* non-manifold half-facet.

        WARNING: this routine requires the sibling half-facet data to be built first.
        """
        non_manifold_hf = [] # init to empty list

        # go thru all the elements
        NC = self.Size()
        for ci in np.arange(0, NC, dtype=CellIndType):
            #const CellSimplex_DIM CL = Get_Cell_struct(ci);
            CL_halffacet = self.halffacet[ci]
            # loop through all the half-facets
            for fi in np.arange(0, self._cell_dim+1, dtype=SmallIndType):
                # get the facet neighbor
                #const HalfFacetType& neighbor_hf = CL.halffacet[fi];
                neighbor_hf = CL_halffacet[fi]
                n_ci = neighbor_hf['ci']
                n_fi = neighbor_hf['fi']

                # if the neighbor is not-NULL
                if (neighbor_hf!=NULL_HalfFacet):
                    #const CellSimplex_DIM N_CL = Get_Cell_struct(n_ci);
                    N_CL_halffacet = self.halffacet[n_ci]
                    
                    # if the neighbor half-facet looks back at the cell we started at
                    if (N_CL_halffacet[n_fi]['ci']==ci):
                        # and if the local facet does *not* match where we started
                        if (N_CL_halffacet[n_fi]['fi']!=fi):
                            # then: two distinct facets of the same cell are joined together!!
                            # this should not happen!
                            print("Error in 'CellSimplexType.Get_Nonmanifold_HalfFacets':")
                            print("      Two facets of the same cell are siblings; this should not happen!")
                            assert(N_CL_halffacet[n_fi]['fi']==fi)
                        # else the local facet matches where we started,
                        #      so the starting half-facet only has one neighbor,
                        #      i.e. it is *manifold* (do nothing).
                    else: # there is more than one neighbor, so this half-facet is *not* manifold
                        # get all of the neighbors (including the starting half-facet)
                        vec_neighbor = self.Get_HalfFacets_Attached_To_HalfFacet(neighbor_hf)

                        # get the neighbor half-facet with the largest cell index
                        max_ci = np.argmax(vec_neighbor[:]['ci'])
                        MAX_hf = vec_neighbor[max_ci]
                        # store that half-facet
                        non_manifold_hf.append(MAX_hf)
                # else there is no neighbor so this half-facet is *manifold* (do nothing)

        # now clean it up by removing duplicate half-facets
        temp_np = np.array(non_manifold_hf, dtype=HalfFacetType)
        non_manifold_hf_np = np.unique(temp_np)
        return non_manifold_hf_np

    def Print_Nonmanifold_HalfFacets(self):
        """Print all non-manifold half-facets in the mesh.
        WARNING: this routine requires the sibling half-facet data to be built first.
        """
        non_manifold_hf = self.Get_Nonmanifold_HalfFacets()

        NUM = non_manifold_hf.size
        if (NUM==0):
            print("There are *no* non-manifold half-facets.")
        else: # there is at least 1
            print("These are all the non-manifold half-facets in the mesh (output: (cell index, local facet index)):")
            for hf_jj in non_manifold_hf:
                print(str(hf_jj))

    def Vtx2Adjacent(self, v_in, ci, fi):
        """Given a global vertex index, cell index, and local facet index, return the *other*
        global vertices in that facet.  If the facet does not contain the given vertex,
        then the output is an array of NULL_Vtx's.
        If Dim() < 2, then the output is an empty array, because facets are either non-existent
        or they contain only one vertex.
        Note: (ci,fi) is a half-facet, where ci is a cell index, and fi is a local facet index.
        """
        if (fi < 0) or (fi > self._cell_dim):
            print("Error: facet index fi is negative or bigger than cell dimension!")
        assert ((fi >= 0) and (fi <= self._cell_dim)), "Facet index is invalid!"
        
        if (self._cell_dim >= 2):
            # get the vertices in the half-facet "fi"
            vtx_in_hf = self.Get_Global_Vertices_In_Facet(self.vtx[ci], fi)
            # now return the vertices in the half-facet, EXCEPT v_in (i.e. the adjacent vertices)
            v_adj = self.Get_Adj_Vertices_In_Facet(vtx_in_hf, v_in)
        else:
            v_adj = np.zeros(0, dtype=VtxIndType)
        return v_adj

    def Get_Local_Vertex_Index_In_Cell(self, vi, cell_vtx):
        """Find the *local* index of the given *global* vertex within the given cell vertices.
        """

        # init to invalid value
        ii = NULL_Small
        for kk in range(self._cell_dim+1):
            if cell_vtx[kk]==vi:
                ii = kk
                break
        return ii

    def Get_Local_Facets_Sharing_Local_Vertex(self, vi):
        """Return the local facet indices that share a given local vertex.
        """
        if (vi < 0) or (vi > self._cell_dim):
            print("Error: index is negative or bigger than cell dimension!")
        assert ((vi >= 0) and (vi <= self._cell_dim)), "Index is invalid!"

        # note: vertex vi is opposite facet vi
        #       so, facet vi does NOT contain vertex vi
        
        # make an array: [0, 1, vi-1, vi+1, ..., Dim()]
        a0 = np.arange(0, vi, dtype=SmallIndType)
        a1 = np.arange(vi+1, self._cell_dim+1, dtype=SmallIndType)
        facet_ind = np.concatenate((a0, a1), axis=None)
        return facet_ind

    def Get_Local_Vertices_Of_Local_Facet(self, fi):
        """Return the local vertices that are attached to a given local facet.
        """
        # hahah, we can reuse this!
        vert = self.Get_Local_Facets_Sharing_Local_Vertex(fi)
        return vert
        
    def Get_Global_Vertices_In_Facet(self, vtx_ind, fi):
        """Given a cell's vertices and a local facet index, return the global vertices
        contained in that facet.
        """
        if (fi < 0) or (fi > self._cell_dim):
            print("Error: facet index fi is negative or bigger than cell dimension!")
        assert ((fi >= 0) and (fi <= self._cell_dim)), "Facet index is invalid!"

        # note: vertex fi is opposite facet fi
        #       so, facet fi does NOT contain vertex fi
        
        # copy array over (except for vertex fi)
        facet_vtx = vtx_ind[np.arange(len(vtx_ind))!=fi]
        return facet_vtx

    def Get_Adj_Vertices_In_Facet(self, fv, vi):
        """Returns the (global) facet vertices in fv (numpy array) that are not equal to vi, where
        vi is also in the facet, i.e. it returns the vertices in the facet that are *adjacent* to vi.
        Note: if vi is not in the facet, then returns an array containing NULL_Vtx's (NULL values).
        Note: if self.Dim()<=1, then returned array has zero length.
        """
        
        # init to invalid value
        ii = NULL_Small
        for kk in range(self._cell_dim):
            if fv[kk]==vi:
                ii = kk
                break
        
        # we found a match
        if (ii!=NULL_Small):
            # copy array over (except for the matching entry)
            adj_vtx = fv[np.arange(len(fv))!=ii]
        else:
            # return NULL value
            adj_vtx = np.full(np.max([self._cell_dim - 1, 0]), NULL_Vtx, dtype=VtxIndType)

        return adj_vtx

    def Get_Vertex_With_Largest_Index_In_Facet(self, cell_vtx, fi):
        """Given the global vertex indices of a cell and local facet index,
        find the vertex index in that facet with the largest index.
        """
        if (fi < 0) or (fi > self._cell_dim):
            print("Error: facet index fi is negative or bigger than cell dimension!")
        assert ((fi >= 0) and (fi <= self._cell_dim)), "Facet index is invalid!"
        
        # note: vertex fi is opposite facet fi (when self._cell_dim > 0)
        #       so, facet fi does NOT contain vertex fi
        
        # only take valid values
        if (self._cell_dim==0):
            # the only option
            MAX_vi = cell_vtx[0]
        else:
            sub_ind = np.arange(0, self._cell_dim+1, dtype=SmallIndType)
            sub_ind = np.delete(sub_ind, fi)
            vtx_sub = cell_vtx[sub_ind]
            MAX_vi  = np.amax(vtx_sub)
        return MAX_vi

    def Adj_Vertices_In_Facet_Equal(self, a, b):
        """Return true if arrays are equal, otherwise false.
        Only the indices [0, 1, ..., self.Dim() - 2] are checked.
        This routine is used when the arrays contain the vertices in a facet of a cell.
        Note: if the arrays have zero length, this returns true.
        """
        if (a.size < self._cell_dim - 1) or (b.size < self._cell_dim - 1):
            print("Error: one of the adjacent vertex arrays is too short!")
        assert ((a.size >= self._cell_dim - 1) and (b.size >= self._cell_dim - 1)), "Adjacent vertex arrays too short!"
        
        if (self._cell_dim > 1):
            return np.array_equal(a[0:self._cell_dim - 1], b[0:self._cell_dim - 1])
        else:
            return True


class VtxCoordType:
    """
    Class for storing array of vertex coordinate data.  Meant to be used by the Mesh class.
    """

    def __init__(self, GEO_DIM, res_buf=0.2):
        """
        GEO_DIM >= 0 is the geometric (ambient) dimension that the points live in.
        res_buf in [0.0, 1.0] is optional, and is for reserving extra space for coordinates.
        """
        if (GEO_DIM<0):
            print("Error: geometric dimension must be non-negative!")
        assert(GEO_DIM>=0)
        if np.rint(GEO_DIM).astype(SmallIndType)!=GEO_DIM:
            print("Error: geometric dimension must be a non-negative integer!")
        assert(np.rint(GEO_DIM).astype(SmallIndType)==GEO_DIM)

        self._geo_dim = GEO_DIM
        # vertex coordinates (initialize)
        self.coord = np.full((1, self._geo_dim), 0.0, dtype=CoordType)
        # actual number of vertices
        self._size = 0
        # amount of extra memory to allocate when re-allocating coord
        #       (number between 0.0 and 1.0).
        self._reserve_buffer = res_buf # e.g. 0.2 means extra 20%

        # flag to indicate if vertex coordinates may be added or modified.
        #  true  = vertex coord can be added, modified
        #  false = the vertex coord cannot be changed!
        self._coord_open = True

    def __str__(self):
        dimcoord = self.coord.shape
        OUT_STR = ("The geometric dimension is: " + str(self._geo_dim) + "\n"
                 + "The number of vertices is: " + str(self._size) + "\n"
                 + "The *reserved* size of vertices is: " + str(self.Capacity()) + "\n")
        return OUT_STR

    def Clear(self):
        """This clears all coordinate data.
         The _size attribute (i.e. number of vertices) is set to zero."""
        del(self.coord)
        self.coord = np.full((1, self._geo_dim), 0.0, dtype=CoordType)
        self._size = 0

    def Open(self):
        """This sets the _coord_open flag to True to indicate that
        the coordinates can be modified."""
        self._coord_open = True

    def Close(self):
        """This sets the _coord_open flag to False to indicate that
        the coordinates cannot be modified."""
        self._coord_open = False

    def Is_Coord_Open(self):
        """This prints and returns whether or not the coordinates are open."""
        if not self._coord_open:
            print("Vertex coordinates are not open for modification!")
            print("     You must first use the 'Open' method.")
        return self._coord_open

    def Size(self):
        """This returns the number of vertices."""
        return self._size

    def Dim(self):
        """This returns the geometric dimension of the points."""
        return self._geo_dim

    def Reserve(self, num_vtx):
        """This reserves enough internal storage to hold the given
        number of vertices.  The actual number of vertices does
        not change. (The _size attribute is unchanged.)
        """
        if not self.Is_Coord_Open():
            return

        dimcoord = self.coord.shape
        
        # compute the size needed
        Desired_Size = np.rint(np.ceil((1.0 + self._reserve_buffer) * num_vtx)).astype(VtxIndType)
        if dimcoord[0] < Desired_Size:
            old_size = dimcoord[0]
            self.coord.resize((Desired_Size, self._geo_dim))
            # put in ZERO values
            self.coord[old_size:Desired_Size][:] = 0.0
        else:
            pass

    def Capacity(self):
        """This returns the reserved number of vertex coordinates.
        (Not the capacity which is usually larger.)
        """
        dimcoord = self.coord.shape
        return dimcoord[0]

    def Append(self, vtx_coord):
        """Append several vertices at once by giving their coordinates (as a numpy array).
        vtx_coord has shape (M,GEO_DIM), where M is the number of vertices.
        """
        if not self.Is_Coord_Open():
            return
        if type(vtx_coord) is not np.ndarray:
            print("Error: input must be a numpy array!")
            return

        dim_vc = vtx_coord.shape
        if len(dim_vc)==1:
            num_vtx = 1
            input_geo_dim = dim_vc[0]
        else:
            num_vtx = dim_vc[0]
            input_geo_dim = dim_vc[1]

        Num_Current_Vtx = self.Size()
        New_Total_Vtx = Num_Current_Vtx + num_vtx
        self.Reserve(New_Total_Vtx)
        
        if input_geo_dim==self._geo_dim:
            self.coord[Num_Current_Vtx:New_Total_Vtx][:] = vtx_coord
            self._size += num_vtx
        else:
            print("Error: geometric dimension of given vertex coordinates is incorrect!")

    def Set(self, *args):
        """Set vertex coordinates.  Two ways to call:
        -Overwrite coordinates of specific vertex:
            two inputs: (vtx_ind, vtx_coord), where
            vtx_ind is a single vertex index that already exists,
            vtx_coord is a numpy array of length self.Dim()
        -Set all vertex coordinates at once:
            one input: vtx_coord, which has shape (M,GEO_DIM), where M is the number of vertices.
        """
        if not self.Is_Coord_Open():
            return
        # decipher inputs
        if len(args)==1:
            # set all vertex coordinates
            vtx_coord = args[0]
            if type(vtx_coord) is not np.ndarray:
                print("Error: vtx_coord input must be a numpy array!")
                return
            dim_vc = vtx_coord.shape
            if len(dim_vc)==1:
                num_vtx = 1
                input_geo_dim = dim_vc[0]
            else:
                num_vtx = dim_vc[0]
                input_geo_dim = dim_vc[1]
        elif len(args)==2:
            # set a specific vertex
            vtx_ind   = args[0]
            vtx_coord = args[1]
            if type(vtx_coord) is not np.ndarray:
                print("Error: vtx_coord input must be a numpy array!")
                return

            input_geo_dim = vtx_coord.size
            if ( (vtx_ind==NULL_Vtx) or (vtx_ind>=self._size) ):
                print("Error: given vertex index is not valid!")
                return
        else:
            print("incorrect number of arguments!")
            return

        if input_geo_dim==self._geo_dim:
            if len(args)==1:
                self.Reserve(num_vtx)
                self.coord[0:num_vtx,:] = vtx_coord
                self._size = num_vtx
                # put in ZERO values for the rest
                self.coord[num_vtx+1:][:] = 0.0
            else:
                self.coord[vtx_ind,:] = vtx_coord
        else:
            print("Error: geometric dimension of given vertex coordinates is incorrect!")

    def Change_Dimension(self, new_geo_dim):
        """This changes the geometric dimension of the vertex coordinates.

        E.g. if new_geo_dim==5, then each vertex will have coordinates like
                     (x_0, x_1, x_2, x_3, x_4).

        If the new geometric dimension > old geometric dimension, then:
            (x_0, ..., x_{old GD}) ---> (x_0, ..., x_{old GD}, 0, ..., 0),
        i.e. an extension.

        If the new geometric dimension < old geometric dimension, then:
            (x_0, ..., x_{new GD}, ..., x_{old GD}) ---> (x_0, ..., x_{new GD}),
        i.e. a projection.

        Input: new_geo_dim: the new geometric dimension of the vertex coordinates.
        """
        if not self.Is_Coord_Open():
            print("The coordinates are not *open* to be changed.  Please Open() them.")
            return

        if (new_geo_dim<0):
            print("Error: new geometric dimension must be non-negative!")
        assert(new_geo_dim>=0)
        if np.rint(new_geo_dim).astype(SmallIndType)!=new_geo_dim:
            print("Error: new geometric dimension must be a non-negative integer!")
        assert(np.rint(new_geo_dim).astype(SmallIndType)==new_geo_dim)

        Num_Rows = self.Capacity()
        Num_Cols_old = self.Dim()
        Num_Cols_new = new_geo_dim
        
        if Num_Cols_old==Num_Cols_new:
            # no need to do anything
            return

        old_coord = np.copy(self.coord)
        self.coord = None
        self.coord = np.full((Num_Rows, Num_Cols_new), 0.0, dtype=CoordType)
        
        if Num_Cols_new > Num_Cols_old:
            self.coord[:,0:Num_Cols_old] = old_coord
        elif Num_Cols_new < Num_Cols_old:
            self.coord = old_coord[:,0:Num_Cols_new]

        # update the dimension
        self._geo_dim = new_geo_dim

    def Init_Coord(self, num_vtx):
        """Allocate given number of vertex coordinates and set all to 0.0
        """
        if not self.Is_Coord_Open():
            return

        num_vtx = np.rint(num_vtx).astype(VtxIndType)

        if num_vtx > 0:
            self.Reserve(num_vtx)
            self.coord[:][:] = 0.0
            self._size = num_vtx
        else:
            pass

    def Reindex_Vertices(self, old_indices, new_indices):
        """Re-index the vertex coordinates:
        old_indices[ii] gets mapped to new_indices[ii], for 0 <= ii < old_indices.size.
        Note: the inputs are numpy arrays.
        """
        if not self.Is_Coord_Open():
            return

        # basic check
        num_indices = old_indices.size
        if (num_indices > self.Size()):
            print("Error in 'VtxCoordType.Reindex_Vertices'!")
            print("    The given list of indices is longer than the number of current vertices.")
            return
        if (num_indices!=new_indices.size):
            print("Error in 'VtxCoordType.Reindex_Vertices'!")
            print("    The given index lists are not of the same length.")
            return

        # find the max old and new vertex indices
        MAX_old_vi = np.amax(old_indices)
        MAX_new_vi = np.amax(new_indices)

        # basic check
        if (MAX_old_vi >= self.Size()):
            print("Error in 'VtxCoordType.Reindex_Vertices'!")
            print("    The maximum old vertex index is greater than the current last vertex index.")
            return

        # copy over
        old_coord = np.copy(self.coord)
        # re-init the coordinate list
        self.Clear() # clear the coordinate data
        self.Init_Coord(MAX_new_vi+1) # need to add +1 because this sets the number of vertices
        
        # now map vertices over
        self.coord[new_indices[:],:] = old_coord[old_indices]

    def Bounding_Box(self, VI=None):
        """Get the bounding box of all the vertex coordinates.
        Output: min and max limits of the coordinates (component-wise).
        example:  if GEO_DIM==3, then
        BB_min[:] = [X_min, Y_min, Z_min], (numpy array),
        BB_max[:] = [X_max, Y_max, Z_max], (numpy array).
        If no input is given, all vertices are accounted for when computing the
        bounding box.  Otherwise, only the vertex indices in the given numpy array
        are considered.
        """
        # initialize the box
        BB_min = np.zeros(self._geo_dim, dtype=CoordType)
        BB_max = np.zeros(self._geo_dim, dtype=CoordType)
        
        # now compute
        if VI is None:
            for ii in np.arange(0, self._geo_dim, dtype=SmallIndType):
                BB_min[ii] = np.amin(self.coord[:,ii])
                BB_max[ii] = np.amax(self.coord[:,ii])
        else:
            if type(VI) is not np.ndarray:
                print("Error: VI input must be a numpy array!")
                return
            for ii in np.arange(0, self._geo_dim, dtype=SmallIndType):
                BB_min[ii] = np.amin(self.coord[VI,ii])
                BB_max[ii] = np.amax(self.coord[VI,ii])

        return BB_min, BB_max

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

    def Print(self, vi=NULL_Vtx, num_digit=3):
        """Print all the vertex coordinate data to the screen. "vi" is the index of a
        specific vertex; if vi=NULL_Vtx, then print all vertex coordinates.
        If no vertex index is given, then print all vertex coordinates.
        Optional second argument (default=3) specifies number of digits after decimal point.
        """
        if (vi==NULL_Vtx):
            # then print all the vertex coordinates
            print("Vertex coordinate data:")
            print("-----------------------")
            print("vtx index: [x_{0}, ..., x_{" + str(self._geo_dim-1) + "}]")
            for vi in range(0, self._size):
                vtx_str = self._get_coord_string(vi, num_digit)
                OUT_str = str(vi) + ": " + vtx_str
                print(OUT_str)
        else:
            # then print coordinates for one vertex
            if (vi >= self._size):
                print("Error: invalid vertex index!")
            else:
                OUT = self._get_coord_string(vi, num_digit)
                print(OUT)
        print("")




    #
