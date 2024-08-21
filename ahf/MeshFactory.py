"""
ahf.MeshFactory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for generating meshes to be used with the SimplexMesh class.

Also, see "BaseSimplexMesh.py" and "SimplexMesh.py" for more explanation.

Copyright (c) 07-24-2024,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

import ahf.SimplexMesh as ahf_SM

class MeshFactory:
    r"""
    Class with various methods for generating "standard" meshes.
    
    Note: 
    """

    # def __init__(self, CELL_DIM, res_buf=0.2):

        # if (CELL_DIM<0):
            # print("Error: cell dimension must be non-negative!")
        # assert(CELL_DIM>=0)
        # if np.rint(CELL_DIM).astype(SmallIndType)!=CELL_DIM:
            # print("Error: cell dimension must be a non-negative integer!")
        # assert(np.rint(CELL_DIM).astype(SmallIndType)==CELL_DIM)
        
        # # estimate of the size to allocate in Vtx2HalfFacets
        # self._estimate_size_Vtx2HalfFacets = 0

    def __init__(self):

        print("")

    def __str__(self):

        OUT_STR = ("This class object has various methods for generating meshes and" + "\n"
                 + "outputting them as a (VtxCoordType, SimplexMesh) object." + "\n" )
        return OUT_STR

    # def Clear(self):
        # """This resets all mesh data."""
        # self.Cell.Clear()
        # self.Vtx2HalfFacets.Clear()
        # self._v2hfs.Clear()
        
        # self._mesh_open = True

    # def Open(self):
        # """This sets the _mesh_open flag to True to indicate that
        # the mesh can be modified."""
        # self._mesh_open = True

    # def Close(self):
        # """This sets the _mesh_open flag to False to indicate that
        # the mesh cannot be modified."""
        # self._mesh_open = False

    def Simplex_Mesh_Of_Rectangle(self, Pll=[0.0, 0.0], Pur=[1.0, 1.0], N0=10, N1=10, UseBCC=False):
        """Generate a simplicial mesh of a 2-D rectangle.

        Inputs: Pll: length 2 array with coordinates of lower left corner of rectangle;
                     default = [0,0].
                Pur: length 2 array with coordinates of upper right corner of rectangle;
                     default = [1,1].
                N0: number of intervals to use along the 0-th axis (x-axis); default = 10.
                N1: number of intervals to use along the 1-th axis (y-axis); default = 10.
                UseBCC: choose between a mesh based on a Cartesian lattice with two triangles
                        per "square" (bisected along diagonal), or a lattice that includes
                        the midpoint of each "square" with four triangles;
                        default = False
        Outputs: VC (VtxCoordType), Mesh (SimplexMesh) object containing the mesh data.
        """
        if (type(Pll[0]) is not float) or (type(Pll[1]) is not float):
            print("Error: Pll must be a 2 length array of floats!")
            return
        if (type(Pur[0]) is not float) or (type(Pur[1]) is not float):
            print("Error: Pur must be a 2 length array of floats!")
            return
        if (N0 <= 0) or (N1 <= 0):
            print("Error: number of points N0, N1 must be > 0!")
            return
        if (Pur[0] <= Pll[0]) or (Pur[1] <= Pll[1]):
            print("Error: Pur[:] must be > Pll[:]!")
            return

        # get number of actual points
        NP0 = N0 + 1
        NP1 = N1 + 1
        
        # Create a grid
        xv = np.linspace(0, 1, NP0, dtype=CoordType)
        yv = np.linspace(0, 1, NP1, dtype=CoordType)
        [XX, YY] = np.meshgrid(xv,yv)
        XV = XX.flatten('F')
        XV.shape = [XV.size, 1]
        YV = YY.flatten('F')
        YV.shape = [YV.size, 1]
        XP = np.hstack((XV, YV))
        # apply scaling and translation for dimensions
        L0 = Pur[0] - Pll[0]
        L1 = Pur[1] - Pll[1]
        XP[:,0] = (L0*XP[:,0]) + Pll[0]
        XP[:,1] = (L1*XP[:,1]) + Pll[1]
        
        # meshgrid flips x and y ordering
        idx = np.arange(NP0*NP1, dtype=VtxIndType)
        idx = np.reshape(idx, (NP1, NP0), order='F')
        #idx = reshape(1:prod([ny,nx]),[ny,nx]);
        # local vertex numbering
        v1 = idx[:-1,:-1]
        v1 = v1.flatten('F')
        v1.shape = [v1.size, 1]
        v2 = idx[1:,:-1]
        v2 = v2.flatten('F')
        v2.shape = [v2.size, 1]
        v3 = idx[:-1,1:]
        v3 = v3.flatten('F')
        v3.shape = [v3.size, 1]
        v4 = idx[1:,1:]
        v4 = v4.flatten('F')
        v4.shape = [v4.size, 1]
        
        if UseBCC:
            # cell dimensions
            Cell_X = (1/N0)
            Cell_Y = (1/N1)
            # apply scaling (translation already accounted for)
            Cell_X = L0*Cell_X
            Cell_Y = L1*Cell_Y

            # create BCC coordinates
            New_BCC = XP[v1[:,0],:]
            New_BCC[:,0] = New_BCC[:,0] + Cell_X/2
            New_BCC[:,1] = New_BCC[:,1] + Cell_Y/2

            Num_Cells = N0*N1
            if (New_BCC.shape[0]!=Num_Cells):
                print("Number of cells does not match number of center vertices.")
                return

            v5 = np.arange(Num_Cells, dtype=VtxIndType) + XP.shape[0]
            v5.shape = [v5.size, 1]
            # v5 is linked to v1~v4
            XP = np.vstack((XP,New_BCC))

            # create the cell connectivity
            R0  = np.hstack((v1,v3,v5))
            R1  = np.hstack((v3,v4,v5))
            R2  = np.hstack((v4,v2,v5))
            R3  = np.hstack((v2,v1,v5))
            TRI = np.vstack((R0,R1,R2,R3))
            
        else:
            # create the cell connectivity
            R0  = np.hstack((v1,v3,v4))
            R1  = np.hstack((v1,v4,v2))
            TRI = np.vstack((R0,R1))
            
        # create the object
        VC = ahf_SM.VtxCoordType(2)
        VC.Set(XP)
        Mesh = ahf_SM.SimplexMesh(2,VC)
        Mesh.Set_Cell(TRI)
        
        # finalize it!
        Mesh.Finalize_Mesh_Connectivity()
        Mesh.Close()

        return VC, Mesh


