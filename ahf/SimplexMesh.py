"""
ahf.SimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for array based half-facet (AHF) data structure.  This is a sub-class
of BaseSimplexMesh that also stores a reference to a VtxCoordType object.
This way, many different meshes, of varying topological dimension, can share
the same vertex (point) coordinates.  This also means that we can implement
various methods in a more general way.

Also, see "BaseSimplexMesh.py" and "SimplexMath.py" for more explanation.

Copyright (c) 01-01-2023,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

import ahf.SimplexMath as sm
from ahf.BaseSimplexMesh import *

class SimplexMesh(BaseSimplexMesh):
    """    
    A class for the array based half-facet (AHF) data structure to store and process meshes.
    Can represent simplex meshes of arbitrary dimension (starting with 0).  This is a
    sub-class of BaseSimplexMesh that also stores a reference to a VtxCoordType object.

    Note: the topological dimension of the mesh cannot be greater than the geometric
    dimension of VtxCoord.
    """

    def __init__(self, CELL_DIM, VtxCoord, res_buf=0.2):
        super().__init__(CELL_DIM, res_buf)

        VCType_Correct = isinstance(VtxCoord, VtxCoordType)
        if not VCType_Correct:
            print("Error: VtxCoord is not of VtxCoordType!")
        assert(VCType_Correct)
        self._Vtx = VtxCoord

    def __str__(self):
        Base_str = super().__str__()
        VC_str = self._Vtx.__str__()
        OUT_STR = Base_str + "\n" + VC_str

        return OUT_STR

    def Clear(self):
        """This resets all mesh data, but not the vertex coordinate data."""
        super().Clear()

    def Affine_Map(self, cell_ind=None):
        """Get the (jacobian) matrix and translation vector for the affine map from the
        'standard' reference simplex to a simplex (cell) in the mesh. This is the usual
        finite element affine map.  Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input: cell_ind: non-negative integer being a cell index.
        Outputs: A is the jacobian matrix of shape (GD,TD) for the given cell;
                 b is the translation vector of shape (GD,1) for the given cell.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Outputs: A contains M jacobian matrices, with shape (M,GD,TD) for the given cells;
                 b contains M translation vectors of shape (M,GD,1) for the given cells.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)
        
        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            A, b = sm.Affine_Map(vtx_coord)
        else:
            # more than one cell
            #M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            
            A, b = sm.Affine_Map(vtx_coord)

        return A, b


    # coordinate conversions






    # void Reference_To_Cartesian(const CellIndType&, const CellIndType*, const PointType*, PointType*);
    # void Cartesian_To_Reference(const CellIndType&, const CellIndType*, const PointType*, PointType*);
    # void Barycentric_To_Reference(const CellIndType&, const PointType*, PointType*);
    # void Reference_To_Barycentric(const CellIndType&, const PointType*, PointType*);
    # void Barycentric_To_Cartesian(const CellIndType&, const CellIndType*, const PointType*, PointType*);
    # void Cartesian_To_Barycentric(const CellIndType&, const CellIndType*, const PointType*, PointType*);

    # // cell quantities
    # void Diameter(const CellIndType&, const CellIndType*, RealType*);
    # void Bounding_Box(const CellIndType&, const CellIndType*, PointType*, PointType*);
    # void Bounding_Box(PointType*, PointType*);
    # void Volume(const CellIndType&, const CellIndType*, RealType*);
    # void Angles(const CellIndType&, const CellIndType*, RealType*);

    # // simplex centers
    # void Barycenter(const CellIndType&, const CellIndType*, PointType*);
    # void Circumcenter(const CellIndType&, const CellIndType*, PointType*, RealType*);
    # void Incenter(const CellIndType&, const CellIndType*, PointType*, RealType*);
    
    # // others
    # void Shape_Regularity(const CellIndType&, const CellIndType*, RealType*);


