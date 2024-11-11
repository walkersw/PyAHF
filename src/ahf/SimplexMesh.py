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
        
        # need to enforce that the geometric dimension >= topological dimension
        Dims_Correct = VtxCoord.Dim() >= CELL_DIM
        if not Dims_Correct:
            print("Error: Dimension of VtxCoord must be >= the cell dimension!")
        assert(Dims_Correct)

        self._Vtx = VtxCoord

    def __str__(self):
        Base_str = super().__str__()
        VC_str = self._Vtx.__str__()
        OUT_STR = Base_str + "\n" + VC_str

        return OUT_STR

    def Clear(self):
        """This resets all mesh data, but not the vertex coordinate data."""
        super().Clear()

    def Open(self):
        """This sets the _mesh_open flag and _Vtx._coord_open flag to True
        to indicate that the mesh can be modified."""
        super().Open()
        self._Vtx.Open()

    def Close(self):
        """This sets the _mesh_open flag and _Vtx._coord_open flag to False
        to indicate that the mesh cannot be modified."""
        super().Close()
        self._Vtx.Close()

    def Is_Mesh_Open(self):
        """This prints and returns whether or not the mesh is open."""
        Cell_Mesh_Open = super().Is_Mesh_Open()
        Vtx_Coord_Open = self._Vtx.Is_Coord_Open()
        M_Open = Cell_Mesh_Open and Vtx_Coord_Open
        
        if not Cell_Mesh_Open:
            print("The Cell Mesh is not open for modification!")
            print("     You must first use the 'Open' method.")
        if not Vtx_Coord_Open:
            print("The Vertex Coordinates are not open for modification!")
            print("     You must first use the 'Open' method.")

        return M_Open

    def Set_Geometric_Dimension(self, new_geo_dim):
        """This changes the geometric dimension of the mesh's vertex coordinates.

        Note: the new geometric dimension must be *greater or equal* to the
              topological dimension = TD.

        E.g. if new_geo_dim==5, then each vertex will have coordinates like
                     (x_0, x_1, x_2, x_3, x_4).

        If the new geometric dimension > old geometric dimension, then:
            (x_0, ..., x_{old GD}) ---> (x_0, ..., x_{old GD}, 0, ..., 0),
        i.e. an extension.

        If the new geometric dimension < old geometric dimension, then:
            (x_0, ..., x_{new GD}, ..., x_{old GD}) ---> (x_0, ..., x_{new GD}),
        i.e. a projection.  Warning: this may cause a tangled mesh!

        Input: new_geo_dim: the new geometric dimension of the vertex coordinates.
        """
        if not self._Vtx.Is_Coord_Open():
            print("The mesh coordinates are not *open* to be changed.  Please Open() them.")
            return

        if (new_geo_dim<0):
            print("Error: new geometric dimension must be non-negative!")
        assert(new_geo_dim>=0)
        if np.rint(new_geo_dim).astype(SmallIndType)!=new_geo_dim:
            print("Error: new geometric dimension must be a non-negative integer!")
        assert(np.rint(new_geo_dim).astype(SmallIndType)==new_geo_dim)

        # need to enforce that the new geometric dimension >= topological dimension
        Dims_Correct = new_geo_dim >= self.Top_Dim()
        if not Dims_Correct:
            print("Error: New geometric dimension must be >= topological dimension!")
        assert(Dims_Correct)

        self._Vtx.Change_Dimension(new_geo_dim)

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


    # Perimeter? (get the free boundary...)

    def Get_Vtx_Based_Orthogonal_Frame(self, arg1=None, frame_type="all", debug=False):
        """Get orthonormal column vectors that describe a frame for a set of given
        vertices.  If the keyword "all" is given, then the orthonormal vectors
        span all of \R^{GD}, where the first TD vectors span an (approximate)
        *tangent* space of the mesh (at each vertex), and the last (GD - TD) vectors
        span the space *orthogonal* to the tangent space (i.e. called the *normal*
        space). The keyword "tangent" only gives the vectors in the tangent space;
        the keyword "normal" only gives the vectors in the normal space.
        Note: TD = topological dimension, GD = ambient dimension.

        Note on the method: this a "vertex-based" tangent space.  It is computed by
        computing an SVD of the edge vectors that are connected to the vertex.  Each
        edge vector is the difference of the coordinates of the tail and head vertex
        of that edge.  In other words, the edge vectors are stacked into a matrix,
        and that matrix is SVD'ed.  We then extract the "edge-weighted" orthogonal
        vectors from that matrix.  The first TD vectors are the tangent vectors;
        the rest are the normal vectors.

        There are three ways to call this function:
        Inputs:         vi: non-negative integer being a specific vertex index.
                frame_type: a string = "all", "tangent", or "normal".  If omitted,
                then default is "all".  "all" gives a complete frame of \R^{GD},
                "tangent" gives the tangent space, and "normal" gives the normal space.
        Output: Ortho: numpy matrix (qD,GD), whose rows are the (unit) basis vectors
                       of the frame.
                Note: "all" ==> qD==GD; "tangent" ==> qD==TD; "normal" ==> qD==GD-TD.
        OR
        Inputs: vi: numpy array (N,) of vertex indices.  If set to None (or omitted),
                    then defaults to vi = [0, 1, 2, ..., N-1],
                    where N is the total number of vertices.
                frame_type: same as above.
        Output: Ortho: numpy array (N,qD,GD), which is a stack of N matrices, each
                of whose rows are the (unit) basis vectors of the frame.
                Note: see above, for qD.
        OR
        Inputs: (Mesh_Edges, Vtx2Edge): as generated from 'Get_Vtx_Edge_Star'
                            frame_type: same as above.
        Output: Ortho: either a numpy matrix (qD,GD) or a numpy array (N,qD,GD),
                which is described above; it depends on whether vi was an array or not.
        """
        if isinstance(arg1, tuple):
            Mesh_Edges = arg1[0]
            Vtx2Edge = arg1[1]
            
            if isinstance(Vtx2Edge, dict):
                keys = Vtx2Edge.keys()
                num_keys = len(keys)
                if num_keys > 1:
                    print("Error: if Vtx2Edge is a dict, then it should only contain 1 key!")
                    return
                is_array = False
                vi = keys[0]
            else:
                is_array = True
        else:
            vi = arg1
            if vi is None:
                vi = self.Cell.Get_Unique_Vertices()

            is_array = False
            if isinstance(vi, np.ndarray):
                is_array = True
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
            Mesh_Edges, Vtx2Edge = self.Get_Vtx_Edge_Star(vi, efficient=is_array)

        TD = self.Top_Dim()
        GD = self._Vtx.Dim()

        if not is_array:
            #
            # compute edge vectors (Num_Edges,GD)
            Edge_Vec = self._Vtx.coord[Mesh_Edges[:]['v1']][:] - self._Vtx.coord[Mesh_Edges[:]['v0']][:]
            Max_Edge_in_Star = Vtx2Edge[vi].size
            Tangent_Star = np.zeros((Max_Edge_in_Star,GD), dtype=CoordType)
            Tangent_Star[:,:] = Edge_Vec[Vtx2Edge[vi][:],:]

            U, S, Vh = np.linalg.svd(Tangent_Star, full_matrices=False)

            if frame_type.lower() == "tangent":
                # the first TD rows of Vh span the tangent space
                Ortho = Vh[0:TD,:]
            elif frame_type.lower() == "normal":
                # the remaining GD-TD rows of Vh span the normal space
                Ortho = Vh[TD:,:]
            else:
                Ortho = Vh

        else:
            # we have an array of vertex indices
            
            # compute edge vectors (Num_Edges,GD)
            Edge_Vec = self._Vtx.coord[Mesh_Edges[:,1]][:] - self._Vtx.coord[Mesh_Edges[:,0]][:]
            Max_Vtx = Vtx2Edge.shape[0]
            Max_Edge_in_Star = Vtx2Edge.shape[1]
            Tangent_Star = np.zeros((Max_Vtx,Max_Edge_in_Star,GD), dtype=CoordType)
            Tangent_Star[:,:,:] = Edge_Vec[Vtx2Edge[:,:],:]

            # stack of SVDs
            U, S, Vh = np.linalg.svd(Tangent_Star, full_matrices=False)
            
            # the first TD rows of Vh[ii,:,:] span the tangent space (for each ii)
            # the remaining GD-TD rows of Vh[ii,:,:] span the normal space (for each ii)
            if frame_type.lower() == "tangent":
                Ortho = Vh[:,0:TD,:]
            elif frame_type.lower() == "normal":
                Ortho = Vh[:,TD:,:]
            else:
                Ortho = Vh

        if debug:
            return Ortho, Edge_Vec, Tangent_Star
        else:
            return Ortho

    def Get_Vtx_Based_Orthogonal_Frame_ALT(self, arg1=None, frame_type="all", svd_with="tangent", debug=False):
        """Get orthonormal column vectors that describe a frame for a set of given
        vertices.  If the keyword "all" is given, then the orthonormal vectors
        span all of \R^{GD}, where the first TD vectors span an (approximate)
        *tangent* space of the mesh (at each vertex), and the last (GD - TD) vectors
        span the space *orthogonal* to the tangent space (i.e. called the *normal*
        space). The keyword "tangent" only gives the vectors in the tangent space;
        the keyword "normal" only gives the vectors in the normal space.
        Note: TD = topological dimension, GD = ambient dimension.

        Description of the method: this is a "vertex-based" tangent/normal space. For each
        vertex, we find the cells attached to that vertex.  We then compute the tangent
        basis, or normal basis depending on user choice, for each of those cells.  Next,
        we weight the basis on each cell by the TD-dim volume of that cell.  Then, for
        each vertex, we collect all those weighted basis vectors into one matrix.  We then
        SVD that matrix and extract the first qD (see below) orthogonal, unit basis vectors
        that effectively approximates the tangent (or normal) space of the mesh at that vertex.
        If the whole frame ("all") is desired, then if the SVD was done with the tangent space,
        then the normal basis is exactly orthogonal to the tangent basis (and vice-versa).
        
        There are three ways to call this function:
        Inputs:         vi: non-negative integer being a specific vertex index.
                frame_type: a string = "all", "tangent", or "normal".  If omitted, then
                            default is "all".  "all" gives a complete frame of \R^{GD},
                           "tangent" gives the tangent space, and "normal" gives
                            the normal space.
                  svd_with: a string = "tangent" or "normal".  If omitted, then
                            default is "tangent".  "tangent" ("normal") means to use the
                            tangent (normal) spaces of the local cells when applying the SVD.
        Output: Ortho: numpy matrix (qD,GD), whose rows are the (unit) basis vectors
                       of the frame.
                Note: "all" ==> qD==GD; ordered with the tangent basis vectors first.
                      "tangent" ==> qD==TD; "normal" ==> qD==GD-TD.
        OR
        Inputs:         vi: numpy array (N,) of vertex indices.  If set to None,
                            or omitted, then defaults to vi = [0, 1, 2, ..., N-1],
                            where N is the total number of vertices.
                frame_type: same as above.
                  svd_with: same as above.
        Output: Ortho: numpy array (N,qD,GD), which is a stack of N matrices, each
                of whose rows are the (unit) basis vectors of the frame.
                Note: see above for qD.
        OR
        Inputs:   Vtx2Cell: as generated from 'Get_Vtx_Cell_Attachments'
                frame_type: same as above.
                  svd_with: same as above.
        Output: Ortho: either a numpy matrix (qD,GD) or a numpy array (N,qD,GD),
                which is described above; it depends on whether vi was an array or not.
        """
        if isinstance(arg1, dict):
            Vtx2Cell = arg1
            
            keys = Vtx2Cell.keys()
            num_keys = len(keys)
            if num_keys > 1:
                print("Error: if Vtx2Cell is a dict, then it should only contain 1 key!")
                return
            is_array = False
            vi = keys[0]
        elif isinstance(arg1, np.ndarray):
            arg1_shape = arg1.shape
            if len(arg1_shape)==1:
                # must be an array of vertex indices
                vi = arg1
                is_array = True
                if not ( np.issubdtype(vi.dtype, np.integer) and (np.amin(vi) >= 0) ):
                    print("Error: vi must be a numpy array of non-negative integers!")
                    return
            elif len(arg1_shape)==2:
                # must be a pre-computed Vtx2Cell (efficiently)
                Vtx2Cell = arg1
                is_array = True
            else:
                print("Error: first input argument is invalid!")
                return
        else:
            is_array = False
            
            vi = arg1
            if vi is None:
                vi = self.Cell.Get_Unique_Vertices()
                is_array = True
            elif type(vi) is int:
                if vi < 0:
                    print("Error: vi must be a non-negative integer!")
                    return
            else:
                print("Error: vi must either be a singleton or numpy array of non-negative integers!")
                return
            Vtx2Cell = self.Get_Vtx_Cell_Attachments(vi, efficient=is_array)

        TD = self.Top_Dim()
        GD = self._Vtx.Dim()

        if not is_array:
            # get the cells attached to the given vertex...
            ci_attached = Vtx2Cell[vi]
            
            vtx_ind = self.Cell.vtx[ci_attached,:]
            vtx_coord = self._Vtx.coord[vtx_ind,:]
            # vtx_coord: a (M,TD+1,GD)
            
            if svd_with.lower() == "tangent":
                # use the tangent basis to SVD
                
                # compute the tangent basis on each cell
                CellBasis = sm.Tangent_Space(vtx_coord)
                qD = TD
                # (M,GD,TD)
            else:
                # use the normal basis to SVD
                
                # compute the tangent basis on each cell
                CellBasis = sm.Normal_Space(vtx_coord)
                qD = GD-TD
                # (M,GD,GD-TD)
            
            # weight the basis functions on each cell by the volume of cell
            # (no need to normalize because it does not affect the SVD...)
            Vol = sm.Volume(vtx_coord)
            # SWW: could normalize for numerical stability...
            num_local_cell = CellBasis.shape[0]
            Vol.shape = [num_local_cell, 1, 1]
            W_CellBasis = Vol*CellBasis
            
            # stack these vectors into a matrix and SVD it
            W_CellBasis_tp = np.transpose(W_CellBasis, (0, 2, 1))
            # (M,qD,GD)
            CB_Star = numpy.reshape(W_CellBasis_tp, (num_local_cell*qD, GD))
            # (M*qD,GD)
            
            # extract the orthogonal frame
            U, S, Vh = np.linalg.svd(CB_Star, full_matrices=False)

            if svd_with.lower() == "tangent":
                if frame_type.lower() == "tangent":
                    # the first TD rows of Vh span the tangent space
                    Ortho = Vh[0:TD,:]
                elif frame_type.lower() == "normal":
                    # the remaining GD-TD rows of Vh span the normal space
                    Ortho = Vh[TD:,:]
                else:
                    Ortho = Vh
            else:
                # we used the normal space to do the SVD
                if frame_type.lower() == "normal":
                    # the first qD rows of Vh span the normal space
                    Ortho = Vh[0:qD,:]
                elif frame_type.lower() == "tangent":
                    # the remaining GD-qD rows of Vh span the tangent space
                    Ortho = Vh[qD:,:]
                else:
                    # swap the order so that tangent basis appears first
                    reorder_ind = np.concatenate(np.arange(qD, GD), np.arange(0, qD))
                    Ortho = Vh[reorder_ind,:]

        else:
            # we have an array of vertex indices
            NC = self.Num_Cell()
            all_cell_coord = self._Vtx.coord[self.Cell.vtx[0:NC,:],:]

            if svd_with.lower() == "tangent":
                # use the tangent basis to SVD
                
                # compute the tangent basis on each cell
                CellBasis = sm.Tangent_Space(all_cell_coord)
                qD = TD
                # (NC,GD,TD)
            else:
                # use the normal basis to SVD
                
                # compute the tangent basis on each cell
                CellBasis = sm.Normal_Space(all_cell_coord)
                qD = GD-TD
                # (NC,GD,GD-TD)

            CellBasis_tp = np.transpose(CellBasis, (0, 2, 1))
            # (NC,qD,GD)
            # print("CellBasis_tp")
            # print(CellBasis_tp)

            # weight the tangent basis functions on each cell by the volume of cell
            # (no need to normalize because it does not affect the SVD...)
            Vol = sm.Volume(all_cell_coord)
            Vol.shape = [NC, 1, 1]
            W_CellBasis_tp = np.zeros((NC+1, qD, GD), dtype=CoordType)
            W_CellBasis_tp[0:NC,:,:] = Vol*CellBasis_tp
            # print("W_CellBasis_tp")
            # print(W_CellBasis_tp)

            # replace NULL_Cell with NC in Vtx2Cell
            # print("NC: " + str(NC))
            # print("Vtx2Cell:")
            # print(Vtx2Cell)
            NULL_indices = np.argwhere(Vtx2Cell==NULL_Cell)
            # print("NULL_indices:")
            # print(NULL_indices)
            Vtx2Cell[NULL_indices[:,0],NULL_indices[:,1]] = NC
            # print("Vtx2Cell:")
            # print(Vtx2Cell)
            
            Max_Vtx = Vtx2Cell.shape[0]
            Max_Cell_in_Star = Vtx2Cell.shape[1]
            CB_Star_temp = np.zeros((Max_Vtx,Max_Cell_in_Star,qD,GD), dtype=CoordType)
            
            CB_Star_temp[:,:,:,:] = W_CellBasis_tp[Vtx2Cell[:,:],:,:]
            # (NV,num_local_cell,qD,GD)
            
            # stack these vectors into a matrix and SVD it
            CB_Star = np.reshape(CB_Star_temp, (Max_Vtx,Max_Cell_in_Star*qD, GD))
            # (NV,num_local_cell*qD,GD)

            # SWW: could normalize for numerical stability...

            # stack of SVDs
            U, S, Vh = np.linalg.svd(CB_Star, full_matrices=False)

            if svd_with.lower() == "tangent":
                # the first TD rows of Vh[ii,:,:] span the tangent space (for each ii)
                # the remaining GD-TD rows of Vh[ii,:,:] span the normal space (for each ii)
                if frame_type.lower() == "tangent":
                    Ortho = Vh[:,0:TD,:]
                elif frame_type.lower() == "normal":
                    Ortho = Vh[:,TD:,:]
                else:
                    Ortho = Vh
            else:
                # we used the normal space to do the SVD
                
                # the first qD rows of Vh[ii,:,:] span the normal space (for each ii)
                # the remaining GD-qD rows of Vh[ii,:,:] span the tangent space (for each ii)
                if frame_type.lower() == "normal":
                    Ortho = Vh[:,0:qD,:]
                elif frame_type.lower() == "tangent":
                    Ortho = Vh[:,qD:,:]
                else:
                    # swap the order so that tangent basis appears first
                    reorder_ind = np.concatenate(np.arange(qD, GD), np.arange(0, qD))
                    Ortho = Vh[:,reorder_ind,:]

        if debug:
            return Ortho, CB_Star
        else:
            return Ortho

    def Get_Vtx_Averaged_Normal_Vector(self, arg1=None):
        """Get an averaged unit normal vector at a set of given vertices.
        Note: TD = topological dimension, GD = ambient dimension.  In order to use
              this routine, GD==TD+1 (i.e. the co-dimension must be 1).

        Description of the method: this is a "vertex-based" normal vector. For each
        vertex, we find the cells attached to that vertex.  We then compute the oriented
        normal vector for each of those cells.  Then, we compute a weighted average of
        those normal vectors to get an approximate vertex normal, where the weights are
        the TD-dim volumes of the cells.  Lastly, we perform a normalization step to
        make the vertex normal vector unit length.
        
        There are three ways to call this function:
        Input:          vi: non-negative integer being a specific vertex index.
        Output: normal_vec: numpy array (GD,), representing the components of the normal.
        OR
        Inputs:         vi: numpy array (N,) of vertex indices.  If set to None,
                            or omitted, then defaults to vi = [0, 1, 2, ..., N-1],
                            where N is the total number of vertices.
        Output: normal_vec: numpy array (N,GD), where the ith row is the normal vector
                            of the ith vertex.
        OR
        Inputs:   Vtx2Cell: as generated from 'Get_Vtx_Cell_Attachments'
        Output: normal_vec: a numpy array of shape (GD,) or (N,GD), which is described
                            above; it depends on whether vi was an array or not.
        """
        if isinstance(arg1, dict):
            Vtx2Cell = arg1
            
            keys = Vtx2Cell.keys()
            num_keys = len(keys)
            if num_keys > 1:
                print("Error: if Vtx2Cell is a dict, then it should only contain 1 key!")
                return
            is_array = False
            vi = keys[0]
        elif isinstance(arg1, np.ndarray):
            arg1_shape = arg1.shape
            if len(arg1_shape)==1:
                # must be an array of vertex indices
                vi = arg1
                is_array = True
                if not ( np.issubdtype(vi.dtype, np.integer) and (np.amin(vi) >= 0) ):
                    print("Error: vi must be a numpy array of non-negative integers!")
                    return
            elif len(arg1_shape)==2:
                # must be a pre-computed Vtx2Cell (efficiently)
                Vtx2Cell = arg1
                is_array = True
            else:
                print("Error: first input argument is invalid!")
                return
        else:
            is_array = False
            
            vi = arg1
            if vi is None:
                vi = self.Cell.Get_Unique_Vertices()
                is_array = True
            elif type(vi) is int:
                if vi < 0:
                    print("Error: vi must be a non-negative integer!")
                    return
            else:
                print("Error: vi must either be a singleton or numpy array of non-negative integers!")
                return
            Vtx2Cell = self.Get_Vtx_Cell_Attachments(vi, efficient=is_array)

        TD = self.Top_Dim()
        GD = self._Vtx.Dim()
        if not (GD==TD+1):
            print("Error: this routine requires GD==TD+1!")
            return

        if not is_array:
            # get the cells attached to the given vertex...
            ci_attached = Vtx2Cell[vi]
            
            vtx_ind = self.Cell.vtx[ci_attached,:]
            vtx_coord = self._Vtx.coord[vtx_ind,:]
            # vtx_coord: a (M,TD+1,GD)
            
            # get the cell normal vectors
            NS = sm.Normal_Space(vtx_coord)
            # (M,GD,1)
            
            # weight the normal vectors on each cell by the volume of cell
            Vol = sm.Volume(vtx_coord)
            sum_Vol = np.sum(Vol)
            normed_Vol = Vol / sum_Vol
            num_local_cell = NS.shape[0]
            normed_Vol.shape = [num_local_cell, 1, 1]
            W_NS = normed_Vol*NS
            W_NS.shape = [num_local_cell, GD]
            
            # compute the weighted average
            normal_vec_tilde = np.sum(W_NS, axis=0)
            normal_vec_tilde.shape = [GD]
            MAG = np.linalg.norm(normal_vec_tilde).item(0)
            normal_vec = normal_vec_tilde / MAG

        else:
            # we have an array of vertex indices

            NC = self.Num_Cell()
            all_cell_coord = self._Vtx.coord[self.Cell.vtx[0:NC,:],:]

            # get the cell normal vectors
            NS = np.zeros((NC+1,GD,1), dtype=CoordType)
            NS[0:NC,:,:] = sm.Normal_Space(all_cell_coord)
            # (NC+1,GD,1)

            # replace NULL_Cell with NC in Vtx2Cell
            NULL_indices = np.argwhere(Vtx2Cell==NULL_Cell)
            Vtx2Cell[NULL_indices[:,0],NULL_indices[:,1]] = NC

            # weight the normal vectors on each cell by the volume of cell
            Vol = sm.Volume(all_cell_coord)
            Vol = np.append(Vol, 0.0)
            Vtx2Cell_Vol = Vol[Vtx2Cell[:,:]]
            # print("Vtx2Cell_Vol.shape:")
            # print(Vtx2Cell_Vol.shape)
            # (NV,num_local_cell)
            Vtx_Star_Vol = np.sum(Vtx2Cell_Vol, axis=1)
            # print("Vtx_Star_Vol.shape:")
            # print(Vtx_Star_Vol.shape)
            Vtx_Star_Vol.shape = [Vtx2Cell.shape[0], 1]
            # print("Vtx_Star_Vol.shape:")
            # print(Vtx_Star_Vol.shape)
            
            #np.set_printoptions(precision=18)
            #print("Vtx_Star_Vol:")
            #print(Vtx_Star_Vol)
            
            Vtx2Cell_Vol_normed = Vtx2Cell_Vol / Vtx_Star_Vol
            # print("Vtx2Cell_Vol_normed.shape:")
            # print(Vtx2Cell_Vol_normed.shape)

            Vtx2Cell_Vol_normed.shape = [Vtx2Cell.shape[0], Vtx2Cell.shape[1], 1, 1]
            
            Vtx_Star_NS = NS[Vtx2Cell[:,:],:,:]
            # (NV,num_local_cell,GD,1)
            
            W_NS = Vtx2Cell_Vol_normed*Vtx_Star_NS
            normal_vec_tilde = np.sum(W_NS, axis=1)
            normal_vec_tilde = np.reshape(normal_vec_tilde, (Vtx2Cell.shape[0], GD))
            
            MAG = np.linalg.norm(normal_vec_tilde, axis=1)
            MAG.shape = [Vtx2Cell.shape[0], 1]
            normal_vec = normal_vec_tilde / MAG
            
        return normal_vec



    def Export_For_VTKwrite(self):
        """This exports the vertex coordinates and mesh connectivity in a form
        that VTKwrite (a Python package) can use.
        
        Outputs: Px, Py, Pz, Mesh_Conn, Cell_Offsets.
        
        You then call VTKwrite.unstructuredGridToVTK(...) with the following arguments:
            (FILE_PATH,
            Px, Py, Pz, connectivity = Mesh_Conn, offsets = Cell_Offsets,
            cell_types = ctype,
            all_cell_data = all_cell_data,
            all_point_data = all_point_data,
            comments = comments),
        where ctype = np.zeros(Cell_Offsets.size), ctype[:] = VtkTriangle.tid,
        all_cell_data = <whatever data is defined on cells in the mesh>,
        all_point_data = <whatever data is defined on points in the mesh>,
        comments = <whatever comments you want>.
        """

        TD = self.Top_Dim()
        GD = self._Vtx.Dim()
        
        Num_Points = self._Vtx.Size()
        Px = np.zeros(Num_Points, dtype=CoordType)
        Py = np.zeros(Num_Points, dtype=CoordType)
        Pz = np.zeros(Num_Points, dtype=CoordType)
        if GD > 0:
            Px[:] = self._Vtx.coord[0:Num_Points,0]
        if GD > 1:
            Py[:] = self._Vtx.coord[0:Num_Points,1]
        if GD > 2:
            Pz[:] = self._Vtx.coord[0:Num_Points,2]

        M = self.Num_Cell()
        Mesh_Conn = self.Cell.vtx[0:M,:].flatten('C')
        
        Cell_Offsets = (TD+1) * np.arange(1, M+1, dtype=CellIndType)
        
        return Px, Py, Pz, Mesh_Conn, Cell_Offsets


# (need a local-to-global mapping sub-routine...)
#
# Remove_Unused_Vertices
