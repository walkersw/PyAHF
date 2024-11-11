"""
ahf.BaseSimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base class for array based half-facet (AHF) data structure to store and
process simplex meshes.

Also, see "Vtx2HalfFacet_Mapping.py" and "BasicClasses.py" for more explanation.

Copyright (c) 07-24-2024,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

#from ahf.Vtx2HalfFacet_Mapping import *
from ahf.BasicClasses import *


class BaseSimplexMesh:
    r"""
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

    def Get_Vertices_Attached_To_Vertex(self, vi):
        """Returns all vertex indices (a numpy array) that are attached to vertex "vi"
        by an edge of the mesh; the array is sorted.
        WARNING: this requires the sibling half-facet data (see self.Cell.halffacet)
        to be built, and self.Vtx2HalfFacets must be completed as well, before this
        method can be used.
        Note: the returned cell_array can be empty.
        """
        
        # get all cells attached to vi
        attached_cells = self.Get_Cells_Attached_To_Vertex(vi)

        # collect all the attached vertices
        all_vertices = np.array([], dtype=VtxIndType) # initialize as an empty array
        for ci in attached_cells:
            cell_vtx = self.Cell.vtx[ci]
            all_vertices = np.concatenate((all_vertices, cell_vtx), axis=0)
            # NOTE: in a simplex, all vertices are attached to each other

        vtx_array = np.setdiff1d(all_vertices, vi)
        return vtx_array

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
        """Returns all cell indices (a numpy array) that are attached to vertex "vi",
        and the array is *not* sorted.
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
            temp_array = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, star_hf_it['ci'])
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
            cell_array = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, star_hf_it['ci'])
            print(str(cell_array[0]), endl="")
            # print it
            for cc in np.arange(1, cell_array.size, dtype=CellIndType):
                print(", " + str(cell_array[cc]), endl="")
            print("")

    def Get_Vtx_Cell_Attachments(self, vi=None, efficient=False):
        """Get a mapping from mesh vertices (indices) to mesh cells (indices).

        Input:        vi: a single non-negative integer (vtx index), or
                          a numpy array (VI,) of vertex indices, or None (default).
                          If set to None, then vi = a numpy array (NV,) of all
                          vertex indices that are referenced by the mesh cells.
               efficient: True or False (default).
        -if efficient==False, then: 
        Output: Vtx2Cell: a dict where Vtx2Cell[vi] = numpy array of cell indices that index
                into self.Cell.vtx[:], where vi is a vertex index referenced by self.Cell.
        -if efficient==True, then:
        Output: Vtx2Cell: a numpy array (N,M) where N-1 is the maximum vertex index
                referenced by self.Cell and M is the maximum number of cells that
                any vertex is attached to in the mesh.  Thus,
                Vtx2Cell[vi,:] = numpy array of indices that index into self.Cell.vtx[:].
                If Vtx2Cell[vi,kk]==NULL_Cell, then that is a dummy cell (to be ignored).
                If vi is not referenced by self.Cell, then Vtx2Cell[vi,:]==NULL_Cell.
        """
        if vi is None:
            vi = self.Cell.Get_Unique_Vertices()

        if isinstance(vi, np.ndarray):
            if not ( np.issubdtype(vi.dtype, np.integer) and (np.amin(vi) >= 0) ):
                print("Error: vi must be a numpy array of non-negative integers!")
                return
        elif type(vi) is int:
            if vi < 0:
                print("Error: vi must be a non-negative integer!")
                return
        else:
            print("Error: vi must either be a singleton or numpy array of non-negative integers!")
            return

        if not efficient:
            # just make a dict!
            #Vtx2Cell = dict.fromkeys(vi)
            # NC = self.Num_Cell()
            # for ci in np.arange(NC, dtype=CellIndType):
                # for local_vtx in np.arange(self.Cell._cell_dim+1, dtype=SmallIndType):
                    # current_vi = self.Cell.vtx[ci][local_vtx]
                    # Vtx2Cell[current_vi] = np.append(Vtx2Cell[current_vi], ci)

            Vtx2Cell = dict.fromkeys(vi)
            #Vtx2Cell = {vk: np.zeros(0, dtype=CellIndType) for vk in vi}
            for ii in vi:
                cell_ind_lv  = np.argwhere(self.Cell.vtx[:][:]==ii)
                cell_indices = np.reshape(cell_ind_lv[:,0], (cell_ind_lv.shape[0], ))
                if cell_indices.size > 0:
                    Vtx2Cell[ii] = cell_indices

        else:
            # output a full numpy array that is more efficient for
            # numerical operations later

            min_vi = np.amin(vi)
            max_vi = np.amax(vi)

            # number of unique vertices referenced by Cell
            #Num_Vtx = self.Cell.Num_Vtx()

            # figure out the maximum number of cells per vertex
            Max_Cell_in_Star = 0
            print("vi:")
            print(vi)
            Vtx_Star_temp = dict.fromkeys(vi)
            #Vtx_Star_temp = {vk: np.zeros(0, dtype=CellIndType) for vk in vi}
            for ii in vi:
                cell_ind_lv  = np.argwhere(self.Cell.vtx[:][:]==ii)
                cell_indices = np.reshape(cell_ind_lv[:,0], (cell_ind_lv.shape[0], ))
                Max_Cell_in_Star = np.max([Max_Cell_in_Star, cell_indices.size])
                if cell_indices.size > 0:
                    Vtx_Star_temp[ii] = cell_indices

            # now remake a more efficient data structure
            Vtx2Cell = np.full((max_vi+1,Max_Cell_in_Star), NULL_Cell, dtype=CellIndType)
            for ii in np.arange(max_vi+1, dtype=VtxIndType):
                Cells_Attached_to_V_ii = Vtx_Star_temp[ii]
                Num_Cells = Cells_Attached_to_V_ii.size
                Vtx2Cell[ii,0:Num_Cells] = Cells_Attached_to_V_ii[:]

        return Vtx2Cell

    def Get_Vtx_Edge_Star(self, vi=None, efficient=False):
        """Get a mapping from mesh vertices (indices) to mesh edges (indices).

        Input:        vi: a single non-negative integer (vtx index), or
                          a numpy array (VI,) of vertex indices, or None (default).
                          If set to None, then vi = a numpy array (NV,) of all
                          vertex indices that are referenced by the mesh cells.
               efficient: True or False (default).
        -if efficient==False, then: 
        Outputs: Mesh_Edges: array of MeshEdgeType (see BasicClasses.Cell.Get_Edges())
                 Vtx2Edge: a dict where Vtx2Edge[vi] = numpy array of indices that index
                 into Mesh_Edges, where vi is a vertex index referenced by self.Cell.
        -if efficient==True, then: 
        Outputs: Mesh_Edges: numpy array (E+1,2), where Mesh_Edges[ee,:] contains the
                 vertex indices for a single edge, ee, in the mesh.  Note that edge index
                 E is a dummy edge index with Mesh_Edges[E,:] = [vv, vv], where
                 vv is the smallest index of vertex referenced by self.Cell (usually, vv==0).
                 Vtx2Edge: a numpy array (N,M) where N-1 is the maximum vertex index
                 referenced by self.Cell and M is the maximum number of edges that
                 any vertex is connected to in the mesh.  Thus,
                 Vtx2Edge[vi,:] = numpy array of indices that index into Mesh_Edges.
                 If Vtx2Edge[vi,kk]==E, then that is a dummy edge that should be ignored.
                 If vi is not referenced by self.Cell, then Vtx2Edge[vi,:]==E.
        """
        if vi is None:
            vi = self.Cell.Get_Unique_Vertices()

        if isinstance(vi, np.ndarray):
            if not ( np.issubdtype(vi.dtype, np.integer) and (np.amin(vi) >= 0) ):
                print("Error: vi must be a numpy array of non-negative integers!")
                return
        elif type(vi) is int:
            if vi < 0:
                print("Error: vi must be a non-negative integer!")
                return
        else:
            print("Error: vi must either be a singleton or numpy array of non-negative integers!")
            return

        if not efficient:
            # get all mesh edges (array of MeshEdgeType)
            Mesh_Edges = self.Cell.Get_Edges()

            # just make a dict!
            Vtx2Edge = dict.fromkeys(vi)
            for ii in vi:
                edge_indices_0 = np.argwhere(Mesh_Edges[:]['v0']==ii)
                edge_indices_1 = np.argwhere(Mesh_Edges[:]['v1']==ii)
                edge_ind = np.vstack((edge_indices_0,edge_indices_1))
                if edge_ind.size > 0:
                    Vtx2Edge[ii] = edge_ind[:,0]

        else:
            # output a full numpy array that is more efficient for
            # numerical operations later

            min_vi = np.amin(vi)
            max_vi = np.amax(vi)

            # number of unique vertices referenced by Cell
            #Num_Vtx = self.Cell.Num_Vtx()

            # get all mesh edges (array of MeshEdgeType)
            EE = self.Cell.Get_Edges()

            # figure out the maximum number of edges per vertex
            Max_Edge_in_Star = 0
            Vtx_Star_temp = dict.fromkeys(vi)
            for ii in vi:
                edge_indices_0 = np.argwhere(EE[:]['v0']==ii)
                edge_indices_1 = np.argwhere(EE[:]['v1']==ii)
                edge_ind = np.vstack((edge_indices_0,edge_indices_1))
                Max_Edge_in_Star = np.max([Max_Edge_in_Star, edge_ind.size])
                if edge_ind.size > 0:
                    Vtx_Star_temp[ii] = edge_ind[:,0]

            # make a pure numpy version of Mesh_Edges that has a dummy edge
            Mesh_Edges = np.zeros((EE.size+1,2), dtype=VtxIndType)
            Mesh_Edges[:-1,0] = EE[:]['v0']
            Mesh_Edges[:-1,1] = EE[:]['v1']

            # set the dummy edge (the last edge)
            Mesh_Edges[-1,0] = min_vi
            Mesh_Edges[-1,1] = min_vi
            dummy_edge_index = Mesh_Edges.shape[0]-1

            # now remake a more efficient data structure
            Vtx2Edge = np.full((max_vi+1,Max_Edge_in_Star), dummy_edge_index, dtype=VtxIndType)
            for ii in np.arange(max_vi+1, dtype=VtxIndType):
                Edges_Attached_to_V_ii = Vtx_Star_temp[ii]
                Num_Edges = Edges_Attached_to_V_ii.size
                Vtx2Edge[ii,0:Num_Edges] = Edges_Attached_to_V_ii[:]

        return Mesh_Edges, Vtx2Edge

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

    # def _Build_Sibling_HalfFacets(self):
        # """Fill in the sibling half-facet data structure.
        # Note: this updates the internal data of "self.Cell.halffacet".
        # Note: need to run '_Finalize_v2hfs' before this.
        # """
        # CELL_DIM = self.Top_Dim()
        # if (not self.Is_Mesh_Open()):
            # return

        # # go thru all the elements
        # NC = self.Num_Cell()
        # for ci in np.arange(0, NC, dtype=CellIndType):
            # # loop through all the facets
            # for ff in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                # # if that hf is uninitialized
                # if (self.Cell.halffacet[ci][ff]==NULL_HalfFacet):
                    # # find the vertex with largest ID in the face ff
                    # MaxVtx = self.Cell.Get_Vertex_With_Largest_Index_In_Facet(self.Cell.vtx[ci], ff)
                    # # get vertices in the facet ff of the current element that is adjacent to MaxVtx
                    # Adj_Vtx = self.Cell.Vtx2Adjacent(MaxVtx, ci, ff) # see below...

                    # # find all half-facets that are attached to MaxVtx
                    # HF_1st, Num_HF = self._v2hfs.Get_Half_Facets(MaxVtx)

                    # # update sibling half-facets in Cell to be a cyclic mapping...
                    # if (Num_HF > 0): 
                        # # then there is at least one half-facet attached to MaxVtx
                        # # keep track of consecutive pairs in cyclic mapping

                        # # Note: all the attached half-facets are attached to MaxVtx.
                        # #       we say a half-facet is *valid* if its adjacent vertices match the
                        # #       adjacent vertices in the facet of the original cell (... see above).

                        # # find the first valid half-facet...
                        # Start = np.zeros(1, dtype=VtxIndType)
                        # Start[0] = HF_1st + Num_HF - 1 # default value
                        
                        # for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                            # # get vertices in the half-facet that is adjacent to MaxVtx
                            # star_hf_it = self._v2hfs.VtxMap[hf_it]
                            # Adj_Vtx_First = self.Cell.Vtx2Adjacent(MaxVtx, star_hf_it['ci'], star_hf_it['fi'])

                            # # if the adjacent facet vertices match, then this half-facet is valid
                            # if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_First, Adj_Vtx)):
                                # # ... and save it
                                # Start[0] = hf_it
                                # break

                        # Current = np.zeros(1, dtype=VtxIndType)
                        # # init Current to Start
                        # Current[0] = Start[0]
                        
                        # # loop through the remaining half-facets
                        # for Next in np.arange(Current[0]+1, HF_1st + Num_HF, dtype=VtxIndType):
                            # # get vertices in the half-facet that is adjacent to MaxVtx
                            # star_Next = self._v2hfs.VtxMap[Next]
                            # Adj_Vtx_Other = self.Cell.Vtx2Adjacent(MaxVtx, star_Next['ci'], star_Next['fi'])

                            # # if the half-facet is valid
                            # if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_Other, Adj_Vtx)):
                                # # in the Current cell and half-facet, write the Next half-facet
                                # star_Current = self._v2hfs.VtxMap[Current[0]]
                                # self.Cell.halffacet[star_Current['ci']][star_Current['fi']] = star_Next[['ci','fi']]
                                # # update Current to Next
                                # Current[0] = Next
                                # star_Current = star_Next

                        # # don't forget to close the cycle:
                        # # if Current is different from Start
                        # if (Current[0]!=Start[0]): # i.e. it cannot refer to itself!
                            # # then in the Current cell and half-facet, write the Start half-facet
                            # star_Start = self._v2hfs.VtxMap[Start[0]]
                            # self.Cell.halffacet[star_Current['ci']][star_Current['fi']] = star_Start[['ci','fi']]

                        # # Note: Current and Start are guaranteed to be valid at this point!
                    # else:
                        # # error check!
                        # print("Error: nothing is attached to the largest index vertex in the facet!")
                        # hf = self._v2hfs.Get_Half_Facet(MaxVtx)
                        # assert(hf!=NULL_HalfFacet) # this should stop the program

        # # now that we no longer need v2hfs, we can delete it
        # self._estimate_size_Vtx2HalfFacets = self._v2hfs.Size() # for estimating size of Vtx2HalfFacets
        # self._v2hfs.Clear()

    def _Build_Sibling_HalfFacets(self):
        """Fill in the sibling half-facet data structure.
        Note: this updates the internal data of "self.Cell.halffacet".
        Note: need to run '_Finalize_v2hfs' before this.
        """
        CELL_DIM = self.Top_Dim()
        if (not self.Is_Mesh_Open()):
            return

        # go thru all the cells
        NC = self.Num_Cell()
        for ci in np.arange(0, NC, dtype=CellIndType):
            # loop through all the facets of the current cell
            for ff in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                # if that hf = (ci,ff) is uninitialized
                if (self.Cell.halffacet[ci][ff]==NULL_HalfFacet):
                    VI_facet = np.sort(self.Cell.Get_Global_Vertices_In_Facet(ci, ff))
                    # find the vertex with largest ID in the face ff
                    MaxVtx = VI_facet[-1]

                    # find all half-facets that are attached to MaxVtx
                    HF_1st, Num_HF = self._v2hfs.Get_Half_Facets(MaxVtx)

                    # update sibling half-facets in Cell to be a cyclic mapping
                    if (Num_HF > 0): 
                        # then there is at least one half-facet attached to MaxVtx
                        last_ci = ci
                        last_ff = ff
                        # loop through all half-facets attached to MaxVtx
                        for ii in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                            vhf_ii = self._v2hfs.VtxMap[ii]
                            hf_ii = vhf_ii[['ci','fi']]
                            hf_ii_ci = hf_ii['ci']
                            hf_ii_fi = hf_ii['fi']
                            # if the facet is != to current main facet
                            if ( (hf_ii_ci != ci) or (hf_ii_fi != ff) ):
                                # then check if vertices in the facet match the main facet
                                VI_other_facet = np.sort(self.Cell.Get_Global_Vertices_In_Facet(hf_ii_ci, hf_ii_fi))
                                facet_vertices_match = np.array_equal(VI_facet, VI_other_facet)
                                if facet_vertices_match:
                                    # we found a sibling half-facet!
                                    self.Cell.halffacet[last_ci][last_ff] = hf_ii
                                    # update to make the cyclic mapping
                                    last_ci = hf_ii_ci
                                    last_ff = hf_ii_fi

                        # don't forget to close the cycle:
                        # there needs to be at least 2 half-facets to do this;
                        # otherwise, the main facet has no siblings
                        never_changed = (last_ci == ci) and (last_ff == ff)
                        if not never_changed: # i.e. it cannot refer to itself!
                            self.Cell.halffacet[last_ci][last_ff] = np.array((ci, ff), dtype=HalfFacetType)

                    else:
                        # error check!
                        print("Error: nothing is attached to the largest index vertex in the facet!")
                        hf = self._v2hfs.Get_Half_Facet(MaxVtx)
                        assert(hf!=NULL_HalfFacet) # this should stop the program

        # now that we no longer need v2hfs, we can delete it
        self._estimate_size_Vtx2HalfFacets = self._v2hfs.Size() # for estimating size of Vtx2HalfFacets
        self._v2hfs.Clear()

    def _Check_Sibling_HalfFacets(self):
        """This is an internal debugging routine to check that the
        sibling half-facet data structure is consistent.
        This routine is slow (double FOR loop over the mesh cells).
        """
        CELL_DIM = self.Top_Dim()

        CONSISTENT = True

        NC = self.Num_Cell()
        
        # check each cell
        for ci in np.arange(0, NC, dtype=CellIndType):
            # loop through all the facets of the current cell
            for fi in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                HF_ci_fi = np.array((ci, fi), dtype=HalfFacetType)
                ci_facet_vtx_ind = np.sort(self.Cell.Get_Global_Vertices_In_Facet(ci, fi))
                HF = self.Cell.halffacet[ci][fi]
                if HF==NULL_HalfFacet:
                    # then there should be no sibling
                    # check the entire mesh to make sure that facet does not appear anywhere else
                    for cj in np.arange(0, NC, dtype=CellIndType):
                        if cj!=ci:
                            # loop through all the facets of cj
                            for fj in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                                cj_facet_vtx_ind = np.sort(self.Cell.Get_Global_Vertices_In_Facet(cj, fj))
                                facets_match = np.array_equal(ci_facet_vtx_ind, cj_facet_vtx_ind)
                                if facets_match:
                                    # they should not match!
                                    CONSISTENT = False
                                    print("Error: these two cells have a matching half-facet:")
                                    print("Cell index: " + str(ci) + ":")
                                    self.Cell.Print(ci)
                                    print("Cell index: " + str(cj) + ":")
                                    self.Cell.Print(cj)
                                    print("     But, the sibling half-facet data in ci says NO!")
                else:
                    # there should be (at least one) sibling half-facet
                    
                    # get the complete cyclic mapping for this half-facet
                    HF_map = self.Cell.Get_HalfFacets_Attached_To_HalfFacet(HF)
                    # make sure HF_ci_fi is in HF_np (the cyclic mapping must return to where we started)
                    HF_ci_fi_is_present = False
                    ki = -1
                    for kk in range(HF_map.size):
                        if HF_ci_fi==HF_map[kk]:
                            HF_ci_fi_is_present = True
                            ki = kk
                            break
                    if not HF_ci_fi_is_present:
                        # something is wrong!
                        CONSISTENT = False
                        print("Error: for this cell:")
                        print("Cell index: " + str(ci) + ":")
                        self.Cell.Print(ci)
                        print("       the half-facet:")
                        print(HF_ci_fi)
                        print("       does not appear in the cyclic map:")
                        print(HF_map)
                    else:
                        # check the entire mesh to make sure it appears consistently,
                        #       and not in other places
                        for cj in np.arange(0, NC, dtype=CellIndType):
                            # loop through all the facets of cj
                            for fj in np.arange(0, CELL_DIM+1, dtype=SmallIndType):
                                cj_facet_vtx_ind = np.sort(self.Cell.Get_Global_Vertices_In_Facet(cj, fj))
                                facets_match = np.array_equal(ci_facet_vtx_ind, cj_facet_vtx_ind)
                                if facets_match:
                                    # then check if its just one of the cyclic map HF's
                                    HF_cj_fj = np.array((cj, fj), dtype=HalfFacetType)
                                    HF_cj_fj_is_present = np.isin(HF_cj_fj, HF_map)
                                    if not HF_cj_fj_is_present:
                                        # then we found another sibling half-facet that we should not have!
                                        CONSISTENT = False
                                        print("Error: these two cells have a matching half-facet:")
                                        print("Cell index: " + str(ci) + ":")
                                        self.Cell.Print(ci)
                                        print("Cell index: " + str(cj) + ":")
                                        self.Cell.Print(cj)
                                        print("     But, that half-facet:")
                                        print(HF_cj_fj)
                                        print("     is not in the cyclic map:")
                                        print(HF_map)

        if CONSISTENT:
            print("The sibling half-facet data is consistent!")
        else:
            print("The sibling half-facet data is NOT consistent!")

        return CONSISTENT

    def _Build_Vtx2HalfFacets(self):
        """Build the final vertex-to-adjacent half-facets data structure.
        Note: this needs '_Build_Sibling_HalfFacets' to have already run.
        In general, a vertex is attached to (or contained in) many half-facets.  But we
        only need to store one of the half-facets (for each vertex), because the rest can
        be found by a simple local search using the Sibling Half-Facets.  If the vertex
        is a NON-manifold vertex, then we need to store more than one half-facet in order
        to be able to easily find *all* half-facets (and all cells) that contain that
        vertex.  The number of half-facets needed (for a single vertex) depends on
        the degree of "non-manifold"-ness of the vertex.
        """
        if not self.Is_Mesh_Open():
            print("In order to run '_Build_Vtx2HalfFacets', the mesh must be open for modification!")
            return

        self.Vtx2HalfFacets.Clear() # start fresh
        # make guess on how much space to allocate
        self.Vtx2HalfFacets.Reserve(self._estimate_size_Vtx2HalfFacets)

        # initialize a dict with keys being vertex indices
        min_vi = self.Cell.Min_Vtx_Index()
        max_vi = self.Cell.Max_Vtx_Index()
        Vtx2Cell = {kk: [] for kk in np.arange(min_vi, max_vi+1, dtype=VtxIndType)}
        # this will keep track of what vertices have been visited and distinct components of the mesh

        CELL_DIM = self.Top_Dim()
        NC = self.Num_Cell()

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
                Global_Vtx = self.Cell.vtx[ci][local_vi] # get global vertex
                ci_is_facet_attached_to_GV = ci in Vtx2Cell[Global_Vtx]
                if not ci_is_facet_attached_to_GV:
                    # we have found a cell (containing vi) that is not facet attached to
                    # previously known cells that are facet attached to vi; i.e.
                    # we have found a new component of the mesh that is attached to vi.
                    
                    # current half-facet is (ci,local_fi)
                    local_fi = lv_to_lf[local_vi]
                    # store this half-facet
                    vhf = np.array((Global_Vtx, ci, local_fi), dtype=VtxHalfFacetType)
                    self.Vtx2HalfFacets.Append(vhf)

                    # get the cells attached to the current vertex, that are also
                    #     facet-connected to the current cell
                    attached_cells = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(Global_Vtx, ci)
                    # record this so we know we have visited this component of the mesh
                    Vtx2Cell[Global_Vtx].extend(list(attached_cells[:]))

        # free up some memory
        del Vtx2Cell

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

    def _Check_Vtx2HalfFacets(self, check_freebdy=False):
        """This is an internal debugging routine to check that the
        Vtx2HalfFacets data structure is consistent.
        Note: if check_freebdy==True, then the sibling half-facet data
        must be formed already.
        """
        CELL_DIM = self.Top_Dim()

        CONSISTENT = True

        array_of_cell_indices = np.zeros(0, dtype=CellIndType)

        # go thru all vertices
        UV = self.Get_Unique_Vertices()
        for vi in UV:
            # get the half-facets attached to the vertex
            HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(vi)
            if (Num_HF==0):
                print("Error: there should be at least one attached half-facet!")
                CONSISTENT = False
            elif (Num_HF==1):
                hf = self.Vtx2HalfFacets.VtxMap[HF_1st]
                VI_cell_facet = np.sort(self.Cell.Get_Global_Vertices_In_Facet(hf['ci'], hf['fi']))
                if not np.isin(vi, VI_cell_facet):
                    print("Error: the attached half-facet does not contain the vertex it is attached to!")
                    print("vi = " + str(vi))
                    print("hf:")
                    print(hf)
                    print("Cell index: " + str(hf['ci']) + ":")
                    self.Cell.Print(hf['ci'])
                    CONSISTENT = False
                # get all cell indices that are facet attached to this vertex/cell
                ci_temp = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, hf['ci'])
                array_of_cell_indices = np.append(array_of_cell_indices, ci_temp)
            else:
                # there is more than one, so is a non-manifold vertex
                ci_temp = [[]] * Num_HF
                # loop through all the attached facets
                for it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                    hf_it = self.Vtx2HalfFacets.VtxMap[it]
                    VI_cell_facet = np.sort(self.Cell.Get_Global_Vertices_In_Facet(hf_it['ci'], hf_it['fi']))
                    if not np.isin(vi, VI_cell_facet):
                        print("Error: the attached half-facet does not contain the vertex it is attached to!")
                        print("vi = " + str(vi))
                        print("hf:")
                        print(hf_it)
                        print("Cell index: " + str(hf_it['ci']) + ":")
                        self.Cell.Print(hf_it['ci'])
                        CONSISTENT = False
                    # get all cell indices that are facet attached to this vertex/cell
                    ci_temp = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, hf_it['ci'])
                    array_of_cell_indices = np.append(array_of_cell_indices, ci_temp)
                # loop through distinct pairs of attached facets
                for ii in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                    for jj in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                        if ii!=jj:
                            hf_ii = self.Vtx2HalfFacets.VtxMap[ii]
                            hf_jj = self.Vtx2HalfFacets.VtxMap[jj]
                            c_ii = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, hf_ii['ci'])
                            c_jj = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(vi, hf_jj['ci'])
                            c_ii_AND_c_jj = np.intersect1d(c_ii, c_jj)
                            if c_ii_AND_c_jj.size > 0:
                                print("Error: the distinct components should not have overlapping cells!")
                                print("vi = " + str(vi))
                                print("hf_ii:")
                                print(hf_ii)
                                print("hf_jj:")
                                print(hf_jj)
                                print("c_ii:")
                                print(c_ii)
                                print("c_jj:")
                                print(c_jj)
                                CONSISTENT = False

        # one final check
        unique_ci = np.unique(array_of_cell_indices)
        min_ci = np.min(unique_ci)
        max_ci = np.max(unique_ci)
        if min_ci != 0:
            print("The minimum cell index is not 0!")
            CONSISTENT = False
        if max_ci != self.Num_Cell()-1:
            print("The maximum cell index is not Num_Cell!")
            CONSISTENT = False

        if check_freebdy:
            # determine if the vertices on the free boundary point to a
            # boundary half-facet
            
            # get the boundary vertices from this
            Bdy_Vtx = self.Cell.Get_FreeBoundary(True)
            for vi in Bdy_Vtx:
                # get the half-facets attached to the vertex
                HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(vi)
                if (Num_HF==0):
                    print("Error: there should be at least one attached half-facet!")
                    CONSISTENT = False
                else:
                    # loop through all the attached facets
                    for it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                        hf_it = self.Vtx2HalfFacets.VtxMap[it]
                        no_sibling = self.Cell.halffacet[hf_it['ci']][hf_it['fi']]==NULL_HalfFacet
                        if not no_sibling:
                            print("Error: this boundary vertex should point to a half-facet on the boundary!")
                            print("hf:")
                            print(hf_it)
                            print("Cell index: " + str(hf_it['ci']) + ":")
                            self.Cell.Print(hf_it['ci'])
                            CONSISTENT = False

        if CONSISTENT:
            print("The vertex-to-half-facet data is consistent!")
        else:
            print("The vertex-to-half-facet data is NOT consistent!")

        return CONSISTENT

    # OLD:
    # def _Build_Vtx2HalfFacets(self):
        # """Build the final vertex-to-adjacent half-facets data structure.
        # Note: this needs '_Build_Sibling_HalfFacets' to have already run.
        # In general, a vertex is attached to (or contained in) many half-facets.  But we
        # only need to store one of the half-facets (for each vertex), because the rest can
        # be found by a simple local search using the Sibling Half-Facets.  If the vertex
        # is a NON-manifold vertex, then we need to store more than one half-facet in order
        # to be able to easily find *all* half-facets (and all cells) that contain that
        # vertex.  The number of half-facets needed (for a single vertex) depends on
        # the degree of "non-manifold"-ness of the vertex.
        # """
        # if not self.Is_Mesh_Open():
            # return

        # self.Vtx2HalfFacets.Clear() # start fresh
        # # make guess on how much space to allocate
        # self.Vtx2HalfFacets.Reserve(self._estimate_size_Vtx2HalfFacets)

        # # allocate a temp variable and initialize to all false,
        # #    because we have not visited any half-facets yet.
        # CELL_DIM = self.Top_Dim()
        # NC = self.Num_Cell()
        # Marked = np.full((NC, (CELL_DIM+1)), False)

        # # store one-to-one mapping from local vertices to local facets
        # lv_to_lf = np.zeros(CELL_DIM+1, dtype=SmallIndType)
        # if (CELL_DIM>=1):
            # for kk in np.arange(0, CELL_DIM, dtype=SmallIndType):
                # lv_to_lf[kk] = kk+1
            # lv_to_lf[CELL_DIM] = 0

        # # loop through each cell
        # for ci in np.arange(0, NC, 1, dtype=CellIndType):
            # # loop through each local vertex of the current cell
            # for local_vi in np.arange(0, (CELL_DIM+1), 1, dtype=SmallIndType):
                # # current half-facet is (ci,local_fi)
                # local_fi = lv_to_lf[local_vi]
                # if not Marked[ci][local_fi]:
                    # # store this half-facet
                    # Global_Vtx = self.Cell.vtx[ci][local_vi] # get global vertex
                    # vhf = np.array((Global_Vtx, ci, local_fi), dtype=VtxHalfFacetType)
                    # self.Vtx2HalfFacets.Append(vhf)
                    # # denote that we have visited it!
                    # Marked[ci][local_fi] = True

                    # # get the cells attached to the current vertex, that are also
                    # #     facet-connected to the current cell
                    # attached_cells = self.Cell.Get_Cells_Attached_And_Facet_Connected_To_Vertex_Cell(Global_Vtx, ci)
                    
                    # # loop thru the attached cells and mark corresponding half-facets as also visited
                    # for ci_hat in attached_cells:
                        # # the local vertex index within ci_hat
                        # local_vi_hat = self.Cell.Get_Local_Vertex_Index_In_Cell(Global_Vtx, self.Cell.vtx[ci_hat])
                        # # corresponding local facet index
                        # local_fi_hat = lv_to_lf[local_vi_hat]
                        # # mark it!
                        # Marked[ci_hat][local_fi_hat] = True

        # # now this data structure is usable!
        # self.Vtx2HalfFacets.Sort()
        
        # self.Vtx2HalfFacets.Print_Half_Facets()

        # # Give border half-facets (i.e. half-facets with no siblings) higher priority.
        # # This allows us to easily identify vertices that are on the boundary of the mesh.

        # # loop through each cell
        # for ci in np.arange(0, NC, 1, dtype=CellIndType):
            # # loop through each local facet of the current cell
            # for local_fi in np.arange(0, (CELL_DIM+1), 1, dtype=SmallIndType):
                # # if this half-facet has no sibling
                # if (self.Cell.halffacet[ci][local_fi]==NULL_HalfFacet):
                    # if (CELL_DIM > 0):
                        # # get the local vertices of the local facet
                        # local_vtx = self.Cell.Get_Local_Vertices_Of_Local_Facet(local_fi)
                        # # for each vertex of the current half-facet
                        # for local_vi in np.arange(0, CELL_DIM, 1, dtype=SmallIndType):
                            # # get the global vertex
                            # global_vi = self.Cell.vtx[ci][ local_vtx[local_vi] ]
                            # print("[global_vi, ci]:")
                            # print(global_vi)
                            # print(ci)

                            # # get the half-facets attached to the vertex
                            # HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(global_vi)
                            # print("HF_1st:")
                            # print(HF_1st)
                            # print(self.Vtx2HalfFacets.VtxMap[HF_1st])

                            # # find the half-facet in the connected component corresponding to ci,
                            # #      and replace it with the half-facet with no sibling.
                            # if (Num_HF==0):
                                # # error!
                                # print("Error: the first part of 'Build_Vtx2HalfFacets' missed this vertex: " \
                                      # + str(global_vi) + ".")
                                # temp_hf = self.Vtx2HalfFacets.Get_Half_Facet(global_vi)
                                # assert(temp_hf!=NULL_HalfFacet) # this should stop the program
                            # elif (Num_HF==1):
                                # # in this case, it is obvious what to replace
                                # star_HF_1st = self.Vtx2HalfFacets.VtxMap[HF_1st]
                                # star_HF_1st['ci'] = ci
                                # star_HF_1st['fi'] = local_fi
                            # else:
                                # # there is more than one connected component,
                                # #    so we need to find the correct one to replace.
                                # print("Is there really more than one connected component?")
                                # for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                                    # star_hf_it = self.Vtx2HalfFacets.VtxMap[hf_it]
                                    # print("Num_HF = " + str(Num_HF))
                                    # print(star_hf_it)
                                    # CONNECTED = self.Cell.Two_Cells_Are_Facet_Connected(global_vi, ci, star_hf_it['ci'])
                                    # if CONNECTED:
                                        # # then replace this one (note: this still points to 
                                        # #      self.Vtx2HalfFacets.VtxMap[hf_it])
                                        # star_hf_it['ci'] = ci
                                        # star_hf_it['fi'] = local_fi
                                        # break # can stop looking
                                # print("Finished!")

        # # Note: we don't have to sort again,
        # #       because the half-facets are ordered by the attached vertex


# add this:
#
# Get_Adjacency_Matrix
# Reorder
