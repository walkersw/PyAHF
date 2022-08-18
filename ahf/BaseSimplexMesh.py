"""
ahf.BaseSimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base class for array based half-facet (AHF) data structure to store and
process simplex meshes.

Also, see "Vtx2HalfFacet_Mapping.py" for more explanation.

Copyright (c) 08-12-2022,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, PointType
from ahf import *

#from ahf.Vtx2HalfFacet_Mapping import *
from ahf.BasicClasses import *



FIX!!!!


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

    def __init__(self,CELL_DIM):

        if (CELL_DIM<0):
            print("Error: cell dimension must be non-negative!")

        # connectivity and sibling half-facet data
        self.Cell = CellSimplexType(CELL_DIM)
        
        # flag to indicate if mesh cells may be added or modified.
        #  true  = cells can be added, modified
        #  false = the mesh cells cannot be changed!
        self._mesh_open = True
        
        # amount of extra memory to allocate when re-allocating
        #    Cell and Vtx2HalfFacets (number between 0.0 and 1.0).
        self._cell_reserve_buffer = 0.2 # extra 20%
        
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
        OUT_STR = ("The topological dimension is: " + str(self.Cell.Dim()) + "\n"
                 + "The number of cells is: " + str(self.Cell.Size()) + "\n"
                 + "The *reserved* size of cells is: " + str(self.Cell.Capacity()) + "\n"
                 + "The size of the Vertex-to-Half-Facet Map is: " + str(self.Vtx2HalfFacets.Size()) + "\n"
                 + "The *reserved* size of the Vertex-to-Half-Facet Map is: " 
                 + str(self.Vtx2HalfFacets.Capacity()) + "\n" )
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

    def Reserve(self, num_C):
        """Allocate memory to hold a mesh of a given size (plus a little)."""
        if not self.Is_Mesh_Open():
            return

        # compute the actual size to allocate for the cells
        Desired_Size = np.rint(np.ceil((1.0 + self._cell_reserve_buffer) * num_C))
        self.Cell.Reserve(Desired_Size)
        # guess on what to reserve for the intermediate data structure
        num_v2hfs = (self.Cell.Dim() + 1) * Num_C
        self.v2hfs.Reserve(num_v2hfs)

    def Append_Cell(self, vtx_ind):
        """Append a single cell by giving its global vertex indices (as an array).
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Append(vtx_ind)

    def Append_Cell_Batch(self, num_cells, vtx_ind):
        """Append several cells at once by giving their global vertex
        indices (as an array).
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Append_Batch(num_cells, vtx_ind)

    def Set_Cell(self, cell_ind, vtx_ind):
        """Set the vertex data for a given cell (that already exists) by giving its
        global vertex indices (as an array).
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Set(cell_ind, vtx_ind)

    def Set_All_Cell(self, num_cells, vtx_ind):
        """Set all cell data at once.
        """
        if not self.Is_Mesh_Open():
            return

        self.Cell.Set_All(num_cells, vtx_ind)

    def Append_Cell_And_Update(self, vtx_ind):
        """Append a single cell to the end of the list, and build the intermediate
        v2hfs structure (incrementally).
        """
        # get the next cell index
        ci = self.Cell.Size() # i.e. the current size
        self.Append_Cell(vtx_ind)

        # now "ci" is the *current* cell index
        self._Append_Half_Facets(ci, vtx_ind)

    def Get_Unique_Vertices(self):
        """Get unique list of vertices.
        """
        if (self._mesh_open):
            unique_vertices = np.unique(self.Cell.vtx)
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

    def Finalize_Mesh_Connectivity(self):
        """Finalize the data structures for determining mesh connectivity, i.e.
        determine neighbors (sibling half-facets), vtx2half-facet mapping, etc.)
        and *close* the mesh.
        """
        # the mesh must be *open* to do this.
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










    def Print_v2hfs(self, vi):
        """Print (multiple) half-facets attached to a given vertex
        (from intermediate data structure).
        """
        print("'_v2hfs':")
        self._v2hfs.Print_Half_Facets(vi)

    def Print_Vtx2HalfFacets(self, vi):
        """Print half-facets attached to a given vertex (for final data structure).
        """
        print("'Vtx2HalfFacets':")
        self.Vtx2HalfFacets.Print_Half_Facets(vi)

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




    # private methods below this line.

    def _Append_Half_Facets(self, ci, vtx_ind):
        """Append half-facets to v2hfs struct.
        ci = cell index, vtx_ind = array of vertex indices of the cell, ci.
        """
        vhf = np.array((ci, NULL_Small), dtype=HalfFacetType)
        for fi in range(self.Cell._cell_dim+1):
            # associate (local #fi) half-facet with the vertex with largest index
            #           within that half-facet
            vhf['fi'] = fi;
            VTX = self.Cell.Get_Vertex_With_Largest_Index_In_Facet(vtx_ind, fi)
            v2hfs.Append(VTX, vhf)

    def _Finalize_v2hfs(self, Build_From_Scratch=True):
        """Store adjacent half-facets to vertices in intermediate data structure.
        """
        if not self.Is_Mesh_Open():
            return

        if (Build_From_Scratch):
            self._v2hfs.Clear() # start fresh
            # record all vertex indices (with duplicates)
            NC = self.Num_Cell()
            self._v2hfs.Reserve((self.Dim()+1) * NC)
            for ci in range(NC):
                _Append_Half_Facets(ci, self.Cell.vtx[ci])

        # don't forget to sort!
        self._v2hfs.Sort()

    def _Build_Sibling_HalfFacets(self):
        """Fill in the sibling half-facet data structure.
        Note: this updates the internal data of "self.Cell".
        """
        CELL_DIM = self.Dim()
        if ( (not self.Is_Mesh_Open()) or (CELL_DIM==0) ):
            return

        # arr_size = np.max([CELL_DIM-1,0])
        # Adj_Vtx       = np.zeros(arr_size, dtype=VtxIndType)
        # Adj_Vtx_First = np.zeros(arr_size, dtype=VtxIndType)
        # Adj_Vtx_Other = np.zeros(arr_size, dtype=VtxIndType)

        # go thru all the elements
        NC = self.Num_Cell()
        for ci in range(NC):
            # loop through all the facets
            for ff in range(CELL_DIM+1):
                # if that hf is uninitialized
                if (self.Cell[ci].halffacet[ff]==NULL_HalfFacet):
                    # find the vertex with largest ID in the face ff
                    MaxVtx = self.Cell.Get_Vertex_With_Largest_Index_In_Facet(self.Cell[ci].vtx, ff);
                    # get vertices in the facet ff of the current element that is adjacent to MaxVtx
                    Adj_Vtx = self.Cell.Vtx2Adjacent(MaxVtx, ci, ff) # see below...

                    # find all half-facets that are attached to MaxVtx
                    # std::pair<std::vector<VtxHalfFacetType>::const_iterator,
                              # std::vector<VtxHalfFacetType>::const_iterator>  RR; // make range variable
                    # const MedIndType Num_HF = v2hfs.Get_Half_Facets(MaxVtx, RR);
                    HF_1st, Num_HF = self._v2hfs.Get_Half_Facets(vi)
                    
                    # update sibling half-facets in Cell to be a cyclic mapping...
                    if (Num_HF > 0): 
                        # then there is at least one half-facet attached to MaxVtx
                        # keep track of consecutive pairs in cyclic mapping
                        Current = np.zeros(1, dtype=VtxIndType)

                        # Note: all the attached half-facets are attached to MaxVtx.
                        #       we say a half-facet is *valid* if its adjacent vertices match the
                        #       adjacent vertices in the facet of the original cell (... see above).

                        # find the first valid half-facet...
                        #std::vector<VtxHalfFacetType>::const_iterator  Start = RR.second-1; // default value
                        Start = HF_1st + Num_HF - 1
                        
                        for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                        #for (std::vector<VtxHalfFacetType>::const_iterator it=RR.first; it!=RR.second; ++it)
                            # get vertices in the half-facet that is adjacent to MaxVtx
                            star_hf_it = self._v2hfs.VtxMap[hf_it]
                            Adj_Vtx_First = self.Cell.Vtx2Adjacent(MaxVtx, star_hf_it['ci'], star_hf_it['fi'])

                            # if the adjacent facet vertices match, then this half-facet is valid
                            if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_First, Adj_Vtx)):
                                # ... and save it
                                Start = hf_it
                                break

                        # init Current to Start
                        Current = Start
                        
                        # loop through the remaining half-facets
                        for Next in np.arange(Current+1, HF_1st + Num_HF, dtype=VtxIndType):
                        #for (Next=Current+1; Next!=RR.second; ++Next)
                            # get vertices in the half-facet that is adjacent to MaxVtx
                            #Vtx2Adjacent(MaxVtx, (*Next).ci, (*Next).fi, Adj_Vtx_Other);
                            star_Next = self._v2hfs.VtxMap[Next]
                            Adj_Vtx_Other = self.Cell.Vtx2Adjacent(MaxVtx, star_Next['ci'], star_Next['fi'])

                            star_Current = self._v2hfs.VtxMap[Current]
                            # if the half-facet is valid
                            if (self.Cell.Adj_Vertices_In_Facet_Equal(Adj_Vtx_Other, Adj_Vtx)):
                                # in the Current cell and half-facet, write the Next half-facet
                                #CellSimplex_DIM& Current_Cell = Get_Cell_struct((*Current).ci);
                                self.Cell.halffacet[star_Current['ci']][star_Current['fi']][['ci','fi']] = star_Next[['ci','fi']]
                                #Current_Cell.halffacet[(*Current).fi].ci = (*Next).ci;
                                #Current_Cell.halffacet[(*Current).fi].fi = (*Next).fi;
                                # update Current to Next
                                Current = Next

                        # don't forget to close the cycle:
                        # if Current is different from Start
                        if (Current!=Start): # i.e. it cannot refer to itself!
                            # then in the Current cell and half-facet, write the Start half-facet
                            #CellSimplex_DIM& Current_Cell = Get_Cell_struct((*Current).ci);
                            #Current_Cell.halffacet[(*Current).fi].ci = (*Start).ci;
                            #Current_Cell.halffacet[(*Current).fi].fi = (*Start).fi;
                            star_Start = self._v2hfs.VtxMap[Start]
                            self.Cell.halffacet[star_Current['ci']][star_Current['fi']][['ci','fi']] = star_Start[['ci','fi']]

                        # Note: Current and Start are guaranteed to be valid at this point!
                    else:
                        # error check!
                        print("Error: nothing is attached to the largest index vertex in the facet!")
                        hf = self._v2hfs.Get_Half_Facet(MaxVtx)
                        assert(hf!=NULL_HalfFacet) # this should stop the program

        # # clear temp variables
        # delete(Adj_Vtx_Other);
        # delete(Adj_Vtx_First);
        # delete(Adj_Vtx);

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
        CELL_DIM = self.Dim()
        NC = self.Num_Cell()
        Marked = np.full((NC, (CELL_DIM+1)), False)

        # store one-to-one mapping from local vertices to local facets
        lv_to_lf = np.zeros(CELL_DIM+1, dtype=SmallIndType)
        if (CELL_DIM==1):
            lv_to_lf[0] = 0
            lv_to_lf[1] = 1
        else:
            for kk in range(CELL_DIM):
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
                    #for (std::vector<CellIndType>::iterator it = attached_cells.begin(); it < attached_cells.end(); ++it)
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
                            # std::pair <std::vector<VtxHalfFacetType>::iterator,
                                       # std::vector<VtxHalfFacetType>::iterator> RR;
                            # const MedIndType Num_HF = Vtx2HalfFacets.Get_Half_Facets(global_vi, RR);
                            HF_1st, Num_HF = self.Vtx2HalfFacets.Get_Half_Facets(global_vi)

                            # find the half-facet in the connected component corresponding to ci,
                            #      and replace it with the half-facet with no sibling.
                            if (Num_HF==0):
                                # error!
                                print("Error: the first part of 'Build_Vtx2HalfFacets' missed this vertex: " \
                                      + str(global_vi) + ".")
                                temp_hf = self.Vtx2HalfFacets.Get_Half_Facet(global_vi)
                                assert(temp_hf!=NULL_HalfFacet) # this should stop the program
                            else if (Num_HF==1):
                                # in this case, it is obvious what to replace
                                # (*(RR.first)).ci = ci;
                                # (*(RR.first)).fi = local_fi;
                                star_HF_1st = self.Vtx2HalfFacets.VtxMap[HF_1st]
                                star_HF_1st['ci'] = ci
                                star_HF_1st['fi'] = local_fi
                                #self.Vtx2HalfFacets.VtxMap[HF_1st][['ci','fi']] = np.array((ci, local_fi), dtype=HalfFacetType)
                            else:
                                # there is more than one connected component,
                                #    so we need to find the correct one to replace.
                                #for (std::vector<VtxHalfFacetType>::iterator hf_it = RR.first; hf_it != RR.second; ++hf_it)
                                for hf_it in np.arange(HF_1st, HF_1st + Num_HF, dtype=VtxIndType):
                                    star_hf_it = self.Vtx2HalfFacets.VtxMap[hf_it]
                                    CONNECTED = self.Cell.Two_Cells_Are_Facet_Connected(global_vi, ci, star_hf_it['ci'])
                                    if CONNECTED:
                                        # then replace this one (note: this still points to self.Vtx2HalfFacets.VtxMap[hf_it])
                                        star_hf_it['ci'] = ci
                                        star_hf_it['fi'] = local_fi
                                        break # can stop looking

        # Note: we don't have to sort again,
        #       because the half-facets are ordered by the attached vertex



    # Note: all methods below this line require ??????


