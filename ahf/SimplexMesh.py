"""
ahf.SimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for array based half-facet (AHF) data structure.  This is a sub-class
of BaseSimplexMesh that also stores a reference to a VtxCoordType object.
This way, many different meshes, of varying topological dimension, can share
the same vertex (point) coordinates.  This also means that we can implement
various methods in a more general way.

Also, see "BaseSimplexMesh.py" and "SimplexMath.py" for more explanation.

Copyright (c) 07-24-2024,  Shawn W. Walker
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

    def Reference_To_Cartesian(self, cell_ind=None, ref_coord=None):
        """Convert reference element coordinates to cartesian coordinates, where the reference
        element is the "standard" reference simplex.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Inputs:  cell_ind: non-negative integer being a cell index.
                ref_coord: a (TD,) numpy array that gives the coordinates of a point in the
                reference simplex.
        Output: (GD,) numpy array that gives the cartesian coordinates of the given point.
        OR
        Inputs: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
                then defaults to cell_ind = [0, 1, 2, ..., N-1],
                where N==M is the total number of cells.
                ref_coord: a (M,TD) numpy array that gives the coordinates of M points in the
                reference simplex.  Note: if cell_ind==None, then M must equal the
                total number of cells.
        Output: (M,GD) numpy array that gives the cartesian coordinates of the given M points.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)
        if type(ref_coord) is not np.ndarray:
            print("Error: ref_coord must be a numpy array!")
            return

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        TD = self.Top_Dim()
        if single_cell:
            dim_rc = ref_coord.shape
            if dim_rc[0]!=TD:
                print("Error: ref_coord must be a numpy array of shape (TD,) if cell_ind is a single int!")
                return
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            cart_coord = sm.Reference_To_Cartesian(vtx_coord, ref_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            dim_rc = ref_coord.shape
            if (dim_rc[0]!=M) or (dim_rc[1]!=TD):
                print("Error: ref_coord must be a numpy array of shape (M,TD) if cell_ind is (M,)!")
                return

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            cart_coord = sm.Reference_To_Cartesian(vtx_coord, ref_coord)

        return cart_coord

    def Cartesian_To_Reference(self, cell_ind=None, cart_coord=None):
        """Convert cartesian coordinates to reference element coordinates, where the reference
        element is the "standard" reference simplex.  Note: a projection is used to make
        sure the points are actually on the simplex.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Inputs:  cell_ind: non-negative integer being a cell index.
               cart_coord: (GD,) numpy array that gives the cartesian coordinates of
               the given point.
        Output: (TD,) numpy array that gives the coordinates of the point in the
                reference simplex.
        OR
        Inputs: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
                then defaults to cell_ind = [0, 1, 2, ..., N-1],
                where N==M is the total number of cells.
                cart_coord: (M,GD) numpy array that gives the cartesian coordinates of
                the given M points.  Note: if cell_ind==None, then M must equal the
                total number of cells.
        Output: (M,TD) numpy array that gives the coordinates of the M points in the
                reference simplex.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)
        if type(cart_coord) is not np.ndarray:
            print("Error: cart_coord must be a numpy array!")
            return

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            dim_cc = cart_coord.shape
            if dim_cc[0]!=GD:
                print("Error: cart_coord must be a numpy array of shape (GD,) if cell_ind is a single int!")
                return
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            ref_coord = sm.Cartesian_To_Reference(vtx_coord, cart_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            dim_cc = cart_coord.shape
            if (dim_cc[0]!=M) or (dim_cc[1]!=GD):
                print("Error: cart_coord must be a numpy array of shape (M,GD) if cell_ind is (M,)!")
                return

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            ref_coord = sm.Cartesian_To_Reference(vtx_coord, cart_coord)

        return ref_coord

    def Barycentric_To_Reference(self, bary_coord):
        """Convert barycentric coordinates to reference element coordinates, where the
        reference element is the "standard" reference simplex.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input: bary_coord: (TD+1,) numpy array that gives the barycentric coordinates of
               the given point.
        Output: (TD,) numpy array that gives the coordinates of the point in the
                reference simplex.
        OR
        Input: bary_coord: (M,TD+1) numpy array that gives the barycentric coordinates of
               the given M points.
        Output: (M,TD) numpy array that gives the coordinates of the M points in the
                reference simplex.
        """

        ref_coord = sm.Barycentric_To_Reference(bary_coord)
        return ref_coord

    def Reference_To_Barycentric(self, ref_coord):
        """Convert reference element coordinates to barycentric coordinates, where the
        reference element is the "standard" reference simplex.
        Note: TD = topological dimension, GD = ambient dimension.
        
        There are two ways to call this function:
        Input: ref_coord: (TD,) numpy array that gives the coordinates of the point in the
               reference simplex.
        Output: (TD+1,) numpy array that gives the barycentric coordinates of the given point.
        OR
        Input: ref_coord: (M,TD) numpy array that gives the coordinates of the M points in the
               reference simplex.
        Output: (M,TD+1) numpy array that gives the barycentric coordinates of the given M points.
        """

        bary_coord = sm.Reference_To_Barycentric(ref_coord)
        return bary_coord

    def Barycentric_To_Cartesian(self, cell_ind=None, bary_coord=None):
        """Convert barycentric (simplex) coordinates to cartesian coordinates.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Inputs:  cell_ind: non-negative integer being a cell index.
               bary_coord: a (TD+1,) numpy array that gives the barycentric
               coordinates of the point w.r.t. the given simplex coordinates.
        Output: (GD,) numpy array that gives the cartesian coordinates of the given point.
        OR
        Inputs: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
                then defaults to cell_ind = [0, 1, 2, ..., N-1],
                where N==M is the total number of cells.
                bary_coord: a (M,TD+1) numpy array that gives the barycentric
                coordinates of M points w.r.t. the corresponding simplex coordinates.
                Note: if cell_ind==None, then M must equal the total number of cells.
        Output: (M,GD) numpy array that gives the cartesian coordinates of the given M points.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)
        if type(bary_coord) is not np.ndarray:
            print("Error: bary_coord must be a numpy array!")
            return

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        TD = self.Top_Dim()
        if single_cell:
            dim_bc = bary_coord.shape
            if dim_bc[0]!=TD+1:
                print("Error: bary_coord must be a numpy array of shape (TD+1,) if cell_ind is a single int!")
                return
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            cart_coord = sm.Barycentric_To_Cartesian(vtx_coord, bary_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            dim_bc = bary_coord.shape
            if (dim_bc[0]!=M) or (dim_bc[1]!=TD+1):
                print("Error: bary_coord must be a numpy array of shape (M,TD+1) if cell_ind is (M,)!")
                return

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            cart_coord = sm.Barycentric_To_Cartesian(vtx_coord, bary_coord)

        return cart_coord

    def Cartesian_To_Barycentric(self, cell_ind=None, cart_coord=None):
        """Convert cartesian coordinates to barycentric (simplex) coordinates.
        Note: a projection is used to make sure the points are actually on the simplex.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Inputs:  cell_ind: non-negative integer being a cell index.
               cart_coord: (GD,) numpy array that gives the cartesian coordinates of
               the given point.
        Output: (TD+1,) numpy array that gives the barycentric coordinates of the
                given point.
        OR
        Inputs: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
                then defaults to cell_ind = [0, 1, 2, ..., N-1],
                where N==M is the total number of cells.
                cart_coord: (M,GD) numpy array that gives the cartesian coordinates of
                the given M points.  Note: if cell_ind==None, then M must equal the
                total number of cells.
        Output: (M,TD+1) numpy array that gives the barycentric coordinates of the
                given M points.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)
        if type(cart_coord) is not np.ndarray:
            print("Error: cart_coord must be a numpy array!")
            return

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            dim_cc = cart_coord.shape
            if dim_cc[0]!=GD:
                print("Error: cart_coord must be a numpy array of shape (GD,) if cell_ind is a single int!")
                return
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            bary_coord = sm.Cartesian_To_Barycentric(vtx_coord, cart_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            dim_cc = cart_coord.shape
            if (dim_cc[0]!=M) or (dim_cc[1]!=GD):
                print("Error: cart_coord must be a numpy array of shape (M,GD) if cell_ind is (M,)!")
                return

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            bary_coord = sm.Cartesian_To_Barycentric(vtx_coord, cart_coord)

        return bary_coord

    def Diameter(self, cell_ind=None):
        """Compute the diameter of simplices in the mesh.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Output: a single float that is the diameter of the queried cell.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Output: (M,) numpy array that gives the diameters of the given M simplices.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            diam_0 = sm.Diameter(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            diam_0 = sm.Diameter(vtx_coord)

        return diam_0

    def Volume(self, cell_ind=None):
        """Compute the volume of simplices in the mesh.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Output: a number that gives the (TD)-dimensional volume of the queried cell.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Output: (M,) numpy array that gives the (TD)-dimensional volumes of the M simplices.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            vol_0 = sm.Volume(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            vol_0 = sm.Volume(vtx_coord)

        return vol_0

    def Angles(self, cell_ind=None):
        """Get the angles (in radians) that the facets of the simplices make with
        respect to each other.
        Note: TD = topological dimension, GD = ambient dimension.

        NOTE: this only works when TD <= 3; the rest is to be implemented.
        SWW: should be able to do general dimensions by projecting to the
             tangent space of the simplex.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Output: Ang: (Q,) numpy array, where Q=TD*(TD+1)/2, where each entry contains
                the angle (radians) between two facets.
        Example:
            TD==0:  Ang = {}
            TD==1:  Ang = {PI}
            TD==2:  Ang = {(0, 1), (0, 2), (1, 2)},
                           where "(i, j)" means the angle between facet i and facet j.
            TD==3:  Ang = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
            TD==4:  Ang = {(0, 1), (0, 2), (0, 3), (0, 4),
                           (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
            etc...
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Output: Ang: (M,Q) numpy array that gives the angles (radians) for each of
                the given M simplices (see above).
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            ang_0 = sm.Angles(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            ang_0 = sm.Angles(vtx_coord)

        return ang_0

    def Barycenter(self, cell_ind=None):
        """Compute the barycenter of simplices in the mesh.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Output: (GD,) numpy array that gives the cartesian coordinates of
                the barycenter of the cell.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Output: (M,GD) numpy array that gives the cartesian coordinates of
                the barycenters of the given M cells.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            BC_0 = sm.Barycenter(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            BC_0 = sm.Barycenter(vtx_coord)

        return BC_0

    def Circumcenter(self, cell_ind=None):
        """Compute the circumcenter and circumradius of simplices in the mesh.
        see:  https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm#Coxeter
        for the method.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Outputs: CB: (TD+1,) numpy array that gives the barycentric coordinates
                 of the circumcenter of the cell;
                 CR: a single number that gives the circumradius.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Outputs: CB: (M,TD+1) numpy array that gives the barycentric coordinates of the
                 circumcenters of the given M cells;
                 CR: (M,) numpy array that gives the corresponding circumradii.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            CB_0, CR_0 = sm.Circumcenter(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            CB_0, CR_0 = sm.Circumcenter(vtx_coord)

        return CB_0, CR_0

    def Incenter(self, cell_ind=None):
        """Compute the incenter and inradius of simplices in the mesh; see:
        "Coincidences of simplex centers and related facial structures" by Edmonds, et al.
        Note: TD = topological dimension, GD = ambient dimension.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Outputs: IB: (TD+1,) numpy array that gives the barycentric coordinates
                 of the incenter of the cell;
                 IR: a single number that gives the inradius.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Outputs: IB: (M,TD+1) numpy array that gives the barycentric coordinates of the
                 incenters of the given M simplices;
                 IR: (M,) numpy array that gives the corresponding inradii.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            IB_0, IR_0 = sm.Incenter(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            IB_0, IR_0 = sm.Incenter(vtx_coord)

        return IB_0, IR_0

    def Shape_Regularity(self, cell_ind=None):
        """Compute the "shape regularity" of simplices in the mesh,
        i.e. this ratio: circumradius / inradius.

        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Output: RATIO: a single number representing the "shape regularity" ratio.
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Output: RATIO: (M,) numpy array that gives the "shape regularity" ratios.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            RATIO_0 = sm.Shape_Regularity(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            RATIO_0 = sm.Shape_Regularity(vtx_coord)

        return RATIO_0

    def Bounding_Box(self, cell_ind=None):
        """Compute the bounding (cartesian) box of simplices in the mesh.
        Note: TD = topological dimension, GD = ambient dimension.
        
        There are two ways to call this function:
        Input:  cell_ind: non-negative integer being a specific cell index.
        Outputs: BB_min, BB_max.  Both are (GD,) numpy arrays that contain the minimum
                 and maximum coordinates of the "box" that contains the cell.
           Example:  if GD==3, then
                 BB_min[:] = [X_min, Y_min, Z_min], (numpy array),
                 BB_max[:] = [X_max, Y_max, Z_max], (numpy array).
        OR
        Input: cell_ind: numpy array (M,) of cell indices.  If set to None (or omitted),
               then defaults to cell_ind = [0, 1, 2, ..., N-1],
               where N==M is the total number of cells.
        Outputs: BB_min, BB_max.  Both are (M,GD) numpy arrays that contain the minimum
                 and maximum coordinates of the "box" that contains the M cells.
        """
        if cell_ind is None:
            cell_ind = np.arange(0, self.Num_Cell(), dtype=CellIndType)

        single_cell = False
        if type(cell_ind) is int:
            single_cell = True
        if (not single_cell) and (type(cell_ind) is not np.ndarray):
            print("Error: input must be a single (non-negative) integer or numpy array!")
            return

        GD = self._Vtx.Dim()
        if single_cell:
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind,:],:]
            BB_min_0, BB_max_0 = sm.Bounding_Box(vtx_coord)
        else:
            # more than one cell
            M = cell_ind.shape[0]
            #TD = self.Top_Dim()
            #GD = self._Vtx.Dim()

            # get the grouped list of vertex coordinates
            vtx_coord = self._Vtx.coord[self.Cell.vtx[cell_ind[:],:],:]
            BB_min_0, BB_max_0 = sm.Bounding_Box(vtx_coord)

        return BB_min_0, BB_max_0


    # Perimeter?


# (need a local-to-global mapping sub-routine...)
#
# Remove_Unused_Vertices
