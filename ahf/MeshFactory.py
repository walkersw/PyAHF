"""
ahf.MeshFactory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for generating meshes to be used with the SimplexMesh class.

Also, see "BaseSimplexMesh.py" and "SimplexMesh.py" for more explanation.

Copyright (c) 07-24-2024,  Shawn W. Walker
"""

FIX!!!!

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

#from ahf.Vtx2HalfFacet_Mapping import *
from ahf.BasicClasses import *


class MeshFactory:
    r"""
    Class for 
    
    Note: 
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


