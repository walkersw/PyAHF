"""
ahf.BaseSimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base class for array based half-facet (AHF) data structure to store and
process simplex meshes.

Also, see "Vtx2HalfFacet_Mapping.py" and "BasicClasses.py" for more explanation.

Copyright (c) 08-18-2022,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

#from ahf.Vtx2HalfFacet_Mapping import *
from ahf.BasicClasses import *


class BaseSimplexMesh:
    """
    Base class for array based half-facet (AHF) data structure to store and process meshes.
    This base class is used in deriving the 0-D, 1-D, 2-D, and 3-D mesh classes,
    as well as arbitrarily higher dimensions (all *simplex* meshes).
    
    Note: no vertex coordinates are stored in this class; this is purely topological.

    Scheme for ordering *local* topological entities
    ------------------------------------------------

    RULE: local "facet" Fi is always *opposite* local vertex Vi.

    DIM=0. Point: (SPECIAL CASE) facets do not exist,
                  so we identify the vertex with itself as a facet.
        V0 +

    i.e. "facet" F0 = V0

    DIM=1. Interval: facets are vertices (points)

        V0 +-------------------+ V1

    i.e. "facet" F0 = V1 (which is opposite V0), and "facet" F1 = V0 (which is opposite V1)

    DIM=2. Triangle: facets are edges (line segments)

        V2 +
           |\
           |  \
           |    \
           |      \
           |        \  E0
        E1 |          \
           |            \
           |              \
           |                \
           |                  \
        V0 +-------------------+ V1
                    E2

    i.e. "facet" F0 = E0 := [V1, V2] (which is opposite V0),
         "facet" F1 = E1 := [V2, V0] (which is opposite V1),
         "facet" F2 = E2 := [V0, V1] (which is opposite V1).

    DIM=3: Tetrahedron: facets are faces (triangles)

                 V3 +
                   /|\
                  / | \
                 |  |  \
                |   |   \
               |    |    \
               |    |  F1 \               F0 (opposite V0)
              | F2  |      \
              |     |       \
             |   V0 +--------+ V2
             |     /      __/
            |    /  F3 __/
            |  /    __/
           | /   __/
          |/  __/
       V1 +--/

    i.e. facet F0 = [V1, V2, V3] (which is opposite V0),
         facet F1 = [V0, V3, V2] (which is opposite V1),
         facet F2 = [V0, V1, V3] (which is opposite V2),
         facet F3 = [V0, V2, V1] (which is opposite V3).

    Higher DIM: the pattern continues...

    EXAMPLE:  Mesh of two triangles.  In this case, a half-facet == half-edge.

        V3 +-------------------+ V2
           |\                  |
           |  \                |
           |    \        T1    |
           |      \            |
           |        \          |
           |          \        |
           |            \      |
           |     T0       \    |
           |                \  |
           |                  \|
        V0 +-------------------+ V1

    Triangle Connectivity and Sibling Half-Facet (Half-Edge) Data Struct:

    triangle |   vertices   |     sibling half-edges
     indices |  V0, V1, V2  |     E0,     E1,      E2
    ---------+--------------+-------------------------
        0    |   0,  1,  3  |  (1,1), (NULL),  (NULL)
        1    |   1,  2,  3  | (NULL),  (0,0),  (NULL)

    where (Ti,Ei) is a half-edge, where Ti is the *neighbor* triangle index, and
    Ei is the local edge index of Ti that correponds to the half-edge. (NULL) means
    there is no neighbor triangle.

    Vertex-to-Half-Edge Data Struct:

      vertex |  adjacent
     indices | half-edge
    ---------+------------
        0    |   (0,2)
        1    |   (1,2)
        2    |   (1,0)
        3    |   (0,1)

    Diagram depicting half-edges:

                   (1,0)
        V3 +-------------------+ V2
           |\                  |
           |  \          T1    |
           |    \              |
           |      \  (1,1)     |
     (0,1) |        \          | (1,2)
           |    (0,0) \        |
           |            \      |
           |              \    |
           |     T0         \  |
           |                  \|
        V0 +-------------------+ V1
                   (0,2)

    Note: in this example, only need one adjacent half-edge because there are no
          non-manifold vertices.  But we do allow for non-manifold vertices!
    """

    def __init__(self, CELL_DIM, res_buf=0.2):

        if (CELL_DIM<0):
            print("Error: cell dimension must be non-negative!")
        assert(CELL_DIM>=0)
        if np.rint(CELL_DIM).astype(SmallIndType)!=CELL_DIM:
            print("Error: cell dimension must be a non-negative integer!")
        assert(np.rint(CELL_DIM).astype(SmallIndType)==CELL_DIM)

        # connectivity and sibling half-facet data
        self.Cell = CellSimplexType(CELL_DIM, res_buf)
        
        # flag to indicate if mesh cells may be added or modified.
        #  true  = cells can be added, modified
        #  false = the mesh cells cannot be changed!
        self._mesh_open = True
        
        # estimate of the size to allocate in Vtx2HalfFacets
        self._estimate_size_Vtx2HalfFacets = 0
        
        # referenced vertices in Cell and (possibly multiple) attached half-facet(s)
        self.Vtx2HalfFacets = Vtx2HalfFacetMap()
        
        # intermediate data structure for building sibling half-facet information
        self._v2hfs = Vtx2HalfFacetMap() # for a given vertex, it references multiple half-facets.
        # Note: this data structure will NOT NECESSARILY store all referenced vertices
        #       in the triangulation.  This is because the vertex with smallest index
        #       will never be referenced (for example).  This is an internal structure that
        #       is only used to construct the sibling half-facet information (stored in Cell).

    def __str__(self):
        if self._mesh_open:
            open_str = "The mesh is open for editing."
        else:
            open_str = "The mesh is currently closed and cannot be modified."

        Cell_cap, Vtx2HF_cap = self.Capacity()
        OUT_STR = ("The topological dimension is: " + str(self.Cell.Dim()) + "\n"
                 + "The number of cells is: " + str(self.Cell.Size()) + "\n"
                 + "The *reserved* size of cells is: " + str(Cell_cap) + "\n"
                 + "The size of the Vertex-to-Half-Facet Map is: " + str(self.Vtx2HalfFacets.Size()) + "\n"
                 + "The *reserved* size of the Vertex-to-Half-Facet Map is: " + str(Vtx2HF_cap) + "\n"
                 + open_str + "\n" )
        return OUT_STR

    def Clear(self):
        """This resets all mesh data."""
        self.Cell.Clear()
        self.Vtx2HalfFacets.Clear()
        self._v2hfs.Clear()
        
        self._mesh_open = True

    def Open(self):
        """This sets the _mesh_open flag to True to indicate that
        the mesh can be modified."""
        self._mesh_open = True

    def Close(self):
        """This sets the _mesh_open flag to False to indicate that
        the mesh cannot be modified."""
        self._mesh_open = False

    def Is_Mesh_Open(self):
        """This prints and returns whether or not the mesh is open."""
        if not self._mesh_open:
            print("Mesh is not open for modification!")
            print("     You must first use the 'Open' method.")
        return self._mesh_open

    def Num_Cell(self):
        """Returns the number of cells in the mesh."""
        return self.Cell.Size()

    def Top_Dim(self):
        """Returns the topological dimension of the mesh."""
        return self.Cell.Dim()

    def Reserve(self, num_cl):
        """Allocate memory to hold a mesh (of cells) of a given size (plus a little)."""
        if not self.Is_Mesh_Open():
            return

        self.Cell.Reserve(num_cl)
        # guess on what to reserve for the intermediate data structure
        num_v2hfs = (self.Cell.Dim() + 1) * num_cl
        self._v2hfs.Reserve(num_v2hfs)

    def Capacity(self):
        """This returns the reserved number of cells,
        and the reserved number of vtx-to-halffacets (in that order)."""
        Cell_cap   = self.Cell.Capacity()
        Vtx2HF_cap = self.Vtx2HalfFacets.Capacity()
        return Cell_cap, Vtx2HF_cap

    def Append_Cell(self, cell_vtx):
        """Append several cells at once by giving their global vertex indices (as a numpy array).
        cell_vtx has shape (M,self.Dim()+1), where M is the number of cells, or
        cell_vtx is a 1-D array of length self.Dim()+1 in the case of one cell.
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Append(cell_vtx)

    def Set_Cell(self, *args):
        """Set cell data.  Two ways to call:
        -Set all cell data at once:
            one input: cell_vtx, which has shape (M,self.Dim()+1), where M is the number of cells.
        -Overwrite cell vertex indices of specific cell:
            two inputs: (cell_ind, cell_vtx), where
            cell_ind is a single cell index that already exists,
            cell_vtx is a numpy array of length self.Dim()+1
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Set(*args)

    def Append_Cell_And_Update(self, cell_vtx):
        """Append a single cell to the end of the list, and build the intermediate
        v2hfs structure (incrementally).
        """
        if cell_vtx.size!=self.Top_Dim()+1:
            print("Error: size of cell_vtx must be " + str(self.Top_Dim()+1) + "!")
            return

        # get the next cell index
        ci = self.Cell.Size() # i.e. the current size
        self.Append_Cell(cell_vtx)

        # now "ci" is the *current* cell index
        self._Append_Half_Facets(ci, cell_vtx)

    def Get_Unique_Vertices(self):
        """Get unique list of vertices.
        """
        if (self._mesh_open):
            unique_vertices = self.Cell.Get_Unique_Vertices()
        else:
            unique_vertices = self.Vtx2HalfFacets.Get_Unique_Vertices()

        return unique_vertices

    def Print_Unique_Vertices(self):
        """Print unique list of vertices to the screen.
        """
        
        unique_vertices = self.Get_Unique_Vertices()
        
        print("Unique list of vertex indices:")
        print(str(unique_vertices[0]), end="")
        for kk in range(1, unique_vertices.size):
            print(", " + str(unique_vertices[kk]), end="")
        print("")

    def Num_Vtx(self):
        """Returns the number of (unique) vertices referenced in self.Cell."""
        uv = self.Get_Unique_Vertices()
        return uv.size

    def Max_Vtx_Index(self):
        """Returns the largest vertex index referenced in self.Cell."""
        if (self._mesh_open):
            max_vi = self.Cell.Max_Vtx_Index()
        else:
            max_vi = self.Vtx2HalfFacets.Max_Vtx_Index()

        return max_vi

    def Finalize_Mesh_Connectivity(self):
        """Finalize the data structures for determining mesh connectivity, i.e.
        determine neighbors (sibling half-facets), vtx2half-facet mapping, etc.)
        and *close* the mesh.
        """
        # the mesh must be *open* to do this:
        if not self.Is_Mesh_Open():
            return

        # this sequence of commands must be used!
        self._Finalize_v2hfs(True) # setup an intermediate structure
        self._Build_Sibling_HalfFacets()
        self._Build_Vtx2HalfFacets()

        # now *close* the mesh to further modification
        self.Close()

    # all public routines below this need the mesh to be finalized to output
    #     correct information, i.e. the mesh should be "closed" and all internal
    #     data structures updated. This is done by building the sibling half-facet
    #     structure, and filling out the Vtx2HalfFacets mapping. All of this is
    #     automatically done by the "Finalize_Mesh_Connectivity" method.

    def Print_Cell(self, ci=NULL_Cell):
        """Print cell connectivity and sibling half-facets. "ci" is the index of a
        specific cell; if ci=NULL_Cell, then print all cells.  If no cell index is
        given, then print all cells.
        """
        print("'Cell':")
        self.Cell.Print(ci)

    def Print_v2hfs(self, vi=NULL_Vtx):
        """Print (multiple) half-facets attached to a given vertex
        (from intermediate data structure).
        """
        print("'_v2hfs':")
        self._v2hfs.Print_Half_Facets(vi)

    def Print_Vtx2HalfFacets(self, vi=NULL_Vtx):
        """Print half-facets attached to a given vertex (for final data structure).
        """
        print("'Vtx2HalfFacets':")
        self.Vtx2HalfFacets.Print_Half_Facets(vi)

    def Print(self):
        """Print cell connectivity, sibling half-facets, and the Vtx2HalfFacets data
        structure.
        """
        self.Print_Cell()
        self.Print_Vtx2HalfFacets()

    def Reindex_Vertices(self, new_indices):
        """Re-index the vertices in the mesh.
        Example: new_index = new_indices[old_index]
        """
        # the mesh must be *open* to do this.
        if not self.Is_Mesh_Open():
            return

        # basic check
        if (new_indices.size < self.Max_Vtx_Index()):
            print("Error in 'BaseSimplexMesh.Reindex_Vertices'!")
            print("    The given list of indices is shorter than the")
            print("    max vertex index referenced by cells in the mesh.")
            return

        self.Cell.Reindex_Vertices(new_indices)
        self.Vtx2HalfFacets.Reindex_Vertices(new_indices)

    def Is_Connected(self, *args):
        """Test if a pair of vertices is connected by an edge; returns True/False.
        Inputs: either give a single MeshEdgeType, or
                give two arguments (tail/head vertex indices of the edge).
        WARNING: this requires the sibling half-facet data (see self.Cell.halffacet)
        to be built, and self.Vtx2HalfFacets must be completed as well, before this
        method can be used.
        """
        if len(args)==1:
            if args[0].dtype!=MeshEdgeType:
                print("Error: input is not a MeshEdgeType!")
            v0 = args[0]['v0']
            v1 = args[0]['v1']
        elif len(args)==2:
            if ( (not isinstance(args[0], int)) or (not isinstance(args[1], int)) ):
                print("Error: two inputs should be integers!")
            v0 = np.array(args[0], dtype=VtxIndType)
            v1 = np.array(args[1], dtype=VtxIndType)
        else:
            print("incorrect number of arguments!")

        if (v0==v1):
            print("Input vertices v0, v1 are the *same*!")
            return True # trivial

        # get all cells attached to v0
        attached_cells = self.Get_Cells_Attached_To_Vertex(v0)
        
        # check if v1 is in any of the cells
        v1_in_cell = False # init
        for ci in attached_cells:
            cell_vtx = self.Cell.vtx[ci]
            v1_in_cell = np.isin(v1, cell_vtx)
            if v1_in_cell:
                break

        return v1_in_cell

    def Get_Cells_Attached_To_Edge(self, *args):
        """Returns all cell indices (numpy array) attached to a given edge.
        Inputs: either give a single MeshEdgeType, or
                give two arguments (tail/head vertex indices of the edge).
        WARNING: this requires the sibling half-facet data (see self.Cell.halffacet)
        to be built, and self.Vtx2HalfFacets must be completed as well, before this
        method can be used.
        """
        if len(args)==1:
            if args[0].dtype!=MeshEdgeType:
                print("Error: input is not a MeshEdgeType!")
            v0 = args[0]['v0']
            v1 = args[0]['v1']
        elif len(args)==2:
            if ( (not isinstance(args[0], int)) or (not isinstance(args[1], int)) ):
                print("Error: two inputs should be integers!")
            v0 = np.array(args[0], dtype=VtxIndType)
            v1 = np.array(args[1], dtype=VtxIndType)
        else:
            print("incorrect number of arguments!")

        if (v0==v1):
            print("Input vertices v0, v1 are the *same*!")

        # get all cells attached to v0
        attached_to_v0 = self.Get_Cells_Attached_To_Vertex(v0)
        # get all cells attached to v1
        attached_to_v1 = self.Get_Cells_Attached_To_Vertex(v1)

        # find the common cell indices
        attached_cells = np.intersect1d(attached_to_v0, attached_to_v1)
        return attached_cells

    def Get_Cells_Attached_To_Vertex(self, vi):
        """Returns all cell indices (a numpy array) that are attached to vertex "vi".
        WARNING: this requires the sibling half-facet data (see self.Cell.halffacet)
        to be built, and self.Vtx2HalfFacets must be completed as well, before this
        method can be used.
        Note: the returned cell_array can be empty.
        """
        cell_array = np.array([], dtype=CellIndType) # initialize as an empty array
        if (vi==NULL_Vtx):
            # the vertex is null, so do nothing.
            return cell_array

        # get the attached half-facets
        HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(vi)

        # loop through each half-facet
        # (each one corresponds to a connected component of the mesh)
        for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
            star_hf_it = self.Vtx2HalfFacets.VtxMap[hf_it]
            temp_array = self.Cell.Get_Cells_Attached_To_Vertex(vi, star_hf_it['ci'])
            # store the found cells in cell_array
            cell_array = np.concatenate((cell_array, temp_array), axis=0)

        return cell_array

    def Print_Cells_Attached_To_Vertex(self, vi):
        """Print all cell indices that are attached to vertex "vi".
        This prints out the separate connected components.
        WARNING: this requires the sibling half-facet data (see self.Cell.halffacet)
        to be built, and self.Vtx2HalfFacets must be completed as well, before this
        method can be used.
        """
        if (vi==NULL_Vtx):
            # the vertex is null, so do nothing.
            print("Vertex is invalid, so print nothing!")
            return

        # get the attached half-facets
        HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(vi)

        print("The following cells are attached to vertex #" + str(vi) + ":")

        # loop through each half-facet
        # (each one corresponds to a connected component of the mesh)
        COUNT = 0
        for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
            star_hf_it = self.Vtx2HalfFacets.VtxMap[hf_it]
            COUNT += 1
            print("component #" + str(COUNT) + ": ")
            cell_array = self.Cell.Get_Cells_Attached_To_Vertex(vi, star_hf_it['ci'])
            print(str(cell_array[0]), endl="")
            # print it
            for cc in np.arange(1, cell_array.size, dtype=CellIndType):
                print(", " + str(cell_array[cc]), endl="")
            print("")

    def Get_Nonmanifold_HalfFacets(self):
        """Get a unique set of non-manifold half-facets. This returns a numpy array
        of half-facets, each defining a *distinct* non-manifold half-facet.

        WARNING: this routine requires the sibling half-facet data to be built first.
        """
        non_manifold_hf = self.Cell.Get_Nonmanifold_HalfFacets()
        return non_manifold_hf

    def Print_Nonmanifold_HalfFacets(self):
        """Print all non-manifold half-facets in the mesh.
        WARNING: this routine requires the sibling half-facet data to be built first.
        """
        self.Cell.Print_Nonmanifold_HalfFacets()

    def Get_Nonmanifold_Vertices(self):
        """Get the set of non-manifold vertices. This returns a numpy array of
        vertex indices, each defining a *distinct* non-manifold vertex.
        WARNING: this requires the 'Vtx2HalfFacets' variable to be filled in and complete.
        """
        non_man_vtx = [] # init to empty list

        # access the vtx-to-half-facet data
        V2HF = self.Vtx2HalfFacets.VtxMap
        
        # check everything (note: V2HF is already sorted.)
        # Note: no need to check the last entry.
        for it in np.arange(0, self.Vtx2HalfFacets.Size()-1, dtype=VtxIndType):
            next_it = (it+1).astype(VtxIndType)
            current_vtx = V2HF[it]['vtx']
            next_vtx    = V2HF[next_it]['vtx']
            # if a vertex shows up more than once, then it is a *non-manifold* vertex
            if (current_vtx==next_vtx):
                # add this vertex to the output array
                non_man_vtx.append(current_vtx)

        # now clean it up by removing duplicate vertices
        temp_np = np.array(non_man_vtx, dtype=VtxIndType)
        non_manifold_vtx = np.unique(temp_np)
        return non_manifold_vtx

    def Print_Nonmanifold_Vertices(self):
        """Get the set of non-manifold vertices. This returns a numpy array of
        vertex indices, each defining a *distinct* non-manifold vertex.
        WARNING: this requires the 'Vtx2HalfFacets' variable to be filled in.
        """
        non_manifold_vtx = self.Get_Nonmanifold_Vertices()
        
        NUM = non_manifold_vtx.size
        if (NUM==0):
            print("There are *no* non-manifold vertices.")
        else: # there is more than 1
            print("These are all the non-manifold vertex indices:")
            for vi in non_manifold_vtx:
                print(str(vi))

    # private methods below this line.

    def _Append_Half_Facets(self, ci, cell_vtx):
        """Append half-facets to v2hfs struct.
        ci = cell index, cell_vtx = array of vertex indices of the cell, ci.
        """
        vhf = np.array((ci, NULL_Small), dtype=HalfFacetType)
        for fi in np.arange(0, self.Cell._cell_dim+1, dtype=SmallIndType):
            # associate (local #fi) half-facet with the vertex with largest index
            #           within that half-facet
            vhf['fi'] = fi
            VTX = self.Cell.Get_Vertex_With_Largest_Index_In_Facet(cell_vtx, fi)
            self._v2hfs.Append(VTX, vhf)

    def _Finalize_v2hfs(self, Build_From_Scratch=True):
        """Store adjacent half-facets to vertices in intermediate data structure.
        """
        if not self.Is_Mesh_Open():
            return

        if (Build_From_Scratch):
            self._v2hfs.Clear() # start fresh
            # record all vertex indices (with duplicates)
            NC = self.Num_Cell()
            self._v2hfs.Reserve((self.Top_Dim()+1) * NC)
            for ci in range(NC):
                self._Append_Half_Facets(ci, self.Cell.vtx[ci])

        # don't forget to sort!
        self._v2hfs.Sort()

    def _Build_Sibling_HalfFacets(self):
        """Fill in the sibling half-facet data structure.
        Note: this updates the internal data of "self.Cell.halffacet".
        """
        CELL_DIM = self.Top_Dim()
        if (not self.Is_Mesh_Open()):
            return

        # go thru all the elements
        NC = self.Num_Cell()
        for ci in np.arange(0, NC, dtype=CellIndType):
            # loop through all the facets
            for ff in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                # if that hf is uninitialized
                if (self.Cell.halffacet[ci][ff]==NULL_HalfFacet):
                    # find the vertex with largest ID in the face ff
                    MaxVtx = self.Cell.Get_Vertex_With_Largest_Index_In_Facet(self.Cell.vtx[ci], ff)
                    # get vertices in the facet ff of the current element that is adjacent to MaxVtx
                    Adj_Vtx = self.Cell.Vtx2Adjacent(MaxVtx, ci, ff) # see below...

                    # find all half-facets that are attached to MaxVtx
                    HF_1st, Num_HF = self._v2hfs.Get_Half_Facets(MaxVtx)

                    # update sibling half-facets in Cell to be a cyclic mapping...
                    if (Num_HF > 0): 
                        # then there is at least one half-facet attached to MaxVtx
                        # keep track of consecutive pairs in cyclic mapping

                        # Note: all the attached half-facets are attached to MaxVtx.
                        #       we say a half-facet is *valid* if its adjacent vertices match the
                        #       adjacent vertices in the facet of the original cell (... see above).

                        # find the first valid half-facet...
                        Start = np.zeros(1, dtype=VtxIndType)
                        Start[0] = HF_1st + Num_HF - 1 # default value
                        
                        for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                            # get vertices in the half-facet that is adjacent to MaxVtx
                            star_hf_it = self._v2hfs.VtxMap[hf_it]
                            Adj_Vtx_First = self.Cell.Vtx2Adjacent(MaxVtx, star_hf_it['ci'], star_hf_it['fi'])

                            # if the adjacent facet vertices match, then this half-facet is valid
                            if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_First, Adj_Vtx)):
                                # ... and save it
                                Start[0] = hf_it
                                break

                        Current = np.zeros(1, dtype=VtxIndType)
                        # init Current to Start
                        Current[0] = Start[0]
                        
                        # loop through the remaining half-facets
                        for Next in np.arange(Current[0]+1, HF_1st + Num_HF, dtype=VtxIndType):
                            # get vertices in the half-facet that is adjacent to MaxVtx
                            star_Next = self._v2hfs.VtxMap[Next]
                            Adj_Vtx_Other = self.Cell.Vtx2Adjacent(MaxVtx, star_Next['ci'], star_Next['fi'])

                            # if the half-facet is valid
                            if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_Other, Adj_Vtx)):
                                # in the Current cell and half-facet, write the Next half-facet
                                star_Current = self._v2hfs.VtxMap[Current[0]]
                                self.Cell.halffacet[star_Current['ci']][star_Current['fi']] = star_Next[['ci','fi']]
                                # update Current to Next
                                Current[0] = Next
                                star_Current = star_Next

                        # don't forget to close the cycle:
                        # if Current is different from Start
                        if (Current[0]!=Start[0]): # i.e. it cannot refer to itself!
                            # then in the Current cell and half-facet, write the Start half-facet
                            star_Start = self._v2hfs.VtxMap[Start[0]]
                            self.Cell.halffacet[star_Current['ci']][star_Current['fi']] = star_Start[['ci','fi']]

                        # Note: Current and Start are guaranteed to be valid at this point!
                    else:
                        # error check!
                        print("Error: nothing is attached to the largest index vertex in the facet!")
                        hf = self._v2hfs.Get_Half_Facet(MaxVtx)
                        assert(hf!=NULL_HalfFacet) # this should stop the program

        # now that we no longer need v2hfs, we can delete it
        self._estimate_size_Vtx2HalfFacets = self._v2hfs.Size() # for estimating size of Vtx2HalfFacets
        self._v2hfs.Clear()

    def _Build_Vtx2HalfFacets(self):
        """Build the final vertex-to-adjacent half-facets data structure.
        In general, a vertex is attached to (or contained in) many half-facets.  But we
        only need to store one of the half-facets (for each vertex), because the rest can
        be found by a simple local search using the Sibling Half-Facets.  If the vertex
        is a NON-manifold vertex, then we need to store more than one half-facet in order
        to be able to easily find *all* half-facets (and all cells) that contain that
        vertex.  The number of half-facets needed (for a single vertex) depends on
        the degree of "non-manifold"-ness of the vertex.
        """
        if not self.Is_Mesh_Open():
            return

        self.Vtx2HalfFacets.Clear() # start fresh
        # make guess on how much space to allocate
        self.Vtx2HalfFacets.Reserve(self._estimate_size_Vtx2HalfFacets)

        # allocate a temp variable and initialize to all false,
        #    because we have not visited any half-facets yet.
        CELL_DIM = self.Top_Dim()
        NC = self.Num_Cell()
        Marked = np.full((NC, (CELL_DIM+1)), False)

        # store one-to-one mapping from local vertices to local facets
        lv_to_lf = np.zeros(CELL_DIM+1, dtype=SmallIndType)
        if (CELL_DIM>=1):
            for kk in np.arange(0, CELL_DIM, dtype=SmallIndType):
                lv_to_lf[kk] = kk+1
            lv_to_lf[CELL_DIM] = 0

        # loop through each cell
        for ci in np.arange(0, NC, 1, dtype=CellIndType):
            # loop through each local vertex of the current cell
            for local_vi in np.arange(0, (CELL_DIM+1), 1, dtype=SmallIndType):
                # current half-facet is (ci,local_fi)
                local_fi = lv_to_lf[local_vi]
                if not Marked[ci][local_fi]:
                    # store this half-facet
                    Global_Vtx = self.Cell.vtx[ci][local_vi] # get global vertex
                    vhf = np.array((Global_Vtx, ci, local_fi), dtype=VtxHalfFacetType)
                    self.Vtx2HalfFacets.Append(vhf)
                    # denote that we have visited it!
                    Marked[ci][local_fi] = True

                    # get the cells attached to the current vertex, that are also
                    #     facet-connected to the current cell
                    attached_cells = self.Cell.Get_Cells_Attached_To_Vertex(Global_Vtx, ci)
                    
                    # loop thru the attached cells and mark corresponding half-facets as also visited
                    for ci_hat in attached_cells:
                        # the local vertex index within ci_hat
                        local_vi_hat = self.Cell.Get_Local_Vertex_Index_In_Cell(Global_Vtx, self.Cell.vtx[ci_hat])
                        # corresponding local facet index
                        local_fi_hat = lv_to_lf[local_vi_hat]
                        # mark it!
                        Marked[ci_hat][local_fi_hat] = True

        # now this data structure is usable!
        self.Vtx2HalfFacets.Sort()

        # Give border half-facets (i.e. half-facets with no siblings) higher priority.
        # This allows us to easily identify vertices that are on the boundary of the mesh.

        # loop through each cell
        for ci in np.arange(0, NC, 1, dtype=CellIndType):
            # loop through each local facet of the current cell
            for local_fi in np.arange(0, (CELL_DIM+1), 1, dtype=SmallIndType):
                # if this half-facet has no sibling
                if (self.Cell.halffacet[ci][local_fi]==NULL_HalfFacet):
                    if (CELL_DIM > 0):
                        # get the local vertices of the local facet
                        local_vtx = self.Cell.Get_Local_Vertices_Of_Local_Facet(local_fi)
                        # for each vertex of the current half-facet
                        for local_vi in np.arange(0, CELL_DIM, 1, dtype=SmallIndType):
                            # get the global vertex
                            global_vi = self.Cell.vtx[ci][ local_vtx[local_vi] ]

                            # get the half-facets attached to the vertex
                            HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(global_vi)

                            # find the half-facet in the connected component corresponding to ci,
                            #      and replace it with the half-facet with no sibling.
                            if (Num_HF==0):
                                # error!
                                print("Error: the first part of 'Build_Vtx2HalfFacets' missed this vertex: " \
                                      + str(global_vi) + ".")
                                temp_hf = self.Vtx2HalfFacets.Get_Half_Facet(global_vi)
                                assert(temp_hf!=NULL_HalfFacet) # this should stop the program
                            elif (Num_HF==1):
                                # in this case, it is obvious what to replace
                                star_HF_1st = self.Vtx2HalfFacets.VtxMap[HF_1st]
                                star_HF_1st['ci'] = ci
                                star_HF_1st['fi'] = local_fi
                            else:
                                # there is more than one connected component,
                                #    so we need to find the correct one to replace.
                                for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                                    star_hf_it = self.Vtx2HalfFacets.VtxMap[hf_it]
                                    CONNECTED = self.Cell.Two_Cells_Are_Facet_Connected(global_vi, ci, star_hf_it['ci'])
                                    if CONNECTED:
                                        # then replace this one (note: this still points to 
                                        #      self.Vtx2HalfFacets.VtxMap[hf_it])
                                        star_hf_it['ci'] = ci
                                        star_hf_it['fi'] = local_fi
                                        break # can stop looking

        # Note: we don't have to sort again,
        #       because the half-facets are ordered by the attached vertex

