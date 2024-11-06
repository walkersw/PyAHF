"""
ahf.MeshFactory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for generating meshes to be used with the SimplexMesh class.

Also, see "BaseSimplexMesh.py" and "SimplexMesh.py" for more explanation.

Copyright (c) 07-24-2024,  Shawn W. Walker
"""

import numpy as np
from scipy.spatial import Delaunay
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

        Inputs: Pll: length 2 array with coordinates of lower-left corner of rectangle;
                     default = [0,0].
                Pur: length 2 array with coordinates of upper-right corner of rectangle;
                     default = [1,1].
                 N0: number of intervals to use along the 0-th axis (x-axis); default = 10.
                 N1: number of intervals to use along the 1-th axis (y-axis); default = 10.
             UseBCC: choose between a mesh based on a Cartesian lattice with two triangles per
                     "square" (bisected along diagonal), or a lattice that includes the
                     midpoint of each "square" with four triangles; default = False.
        Outputs: VC (VtxCoordType), Mesh (SimplexMesh) object containing the mesh data.
        """
        if (type(Pll[0]) is not float) or (type(Pll[1]) is not float):
            print("Error: Pll must be a 2 length array of floats!")
            return
        if (type(Pur[0]) is not float) or (type(Pur[1]) is not float):
            print("Error: Pur must be a 2 length array of floats!")
            return
        if (N0 <= 0) or (N1 <= 0):
            print("Error: number of intervals N0, N1 must be > 0!")
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

    def Simplex_Mesh_Of_Box(self, Pbll=[0.0, 0.0, 0.0], Ptur=[1.0, 1.0, 1.0],
                                  N0=10, N1=10, N2=10, UseBCC=False, FlatSides=False):
        """Generate a simplicial mesh of a 3-D box (hexahedron).

        Inputs: Pbll: length 3 array with coordinates of bottom-lower-left corner of box;
                      default = [0,0,0].
                Ptur: length 3 array with coordinates of top-upper-right corner of box;
                      default = [1,1,1].
                  N0: number of intervals to use along the 0-th axis (x-axis); default = 10.
                  N1: number of intervals to use along the 1-th axis (y-axis); default = 10.
                  N2: number of intervals to use along the 2-nd axis (z-axis); default = 10.
              UseBCC: choose between a mesh based on a Cartesian lattice with six tetrahedra per
                      "cube" (bisected along diagonal), or a lattice that includes the
                      midpoint of each "cube" with four tetrahedra surrounding every lattice edge;
                      default = False.
           FlatSides: this only matters if UseBCC==True. If FlatSides==False, then the BCC mesh
                      of the "cube" has a ragged boundary (e.g. egg carton).  If FlatSides==True,
                      then, the ragged sides are projected to the exact "cube", so that the sides
                      are actually flat.  In other words, a true "cube" is meshed.
                      Default = False.
                      Note: the BCC lattice is not quite respected at the boundary, but this is ok.
        Outputs: VC (VtxCoordType), Mesh (SimplexMesh) object containing the mesh data.
        """
        if (type(Pbll[0]) is not float) or (type(Pbll[1]) is not float) or (type(Pbll[2]) is not float):
            print("Error: Pbll must be a 3 length array of floats!")
            return
        if (type(Ptur[0]) is not float) or (type(Ptur[1]) is not float) or (type(Ptur[2]) is not float):
            print("Error: Ptur must be a 3 length array of floats!")
            return
        if (N0 <= 0) or (N1 <= 0) or (N2 <= 0):
            print("Error: number of intervals N0, N1, N2 must be > 0!")
            return
        if (Ptur[0] <= Pbll[0]) or (Ptur[1] <= Pbll[1]) or (Ptur[2] <= Pbll[2]):
            print("Error: Ptur[:] must be > Pbll[:]!")
            return

        # get number of actual points
        NP0 = N0 + 1
        NP1 = N1 + 1
        NP2 = N2 + 1

        # Create a grid
        xv = np.linspace(0, 1, NP0, dtype=CoordType)
        yv = np.linspace(0, 1, NP1, dtype=CoordType)
        zv = np.linspace(0, 1, NP2, dtype=CoordType)
        [XX, YY, ZZ] = np.meshgrid(xv,yv,zv)
        XV = XX.flatten('F')
        XV.shape = [XV.size, 1]
        YV = YY.flatten('F')
        YV.shape = [YV.size, 1]
        ZV = ZZ.flatten('F')
        ZV.shape = [ZV.size, 1]
        XP = np.hstack((XV, YV, ZV))
        # apply scaling and translation for dimensions
        L0 = Ptur[0] - Pbll[0]
        L1 = Ptur[1] - Pbll[1]
        L2 = Ptur[2] - Pbll[2]
        XP[:,0] = (L0*XP[:,0]) + Pbll[0]
        XP[:,1] = (L1*XP[:,1]) + Pbll[1]
        XP[:,2] = (L2*XP[:,2]) + Pbll[2]

        # meshgrid flips x and y ordering
        idx = np.arange(NP0*NP1*NP2, dtype=VtxIndType)
        idx = np.reshape(idx, (NP1, NP0, NP2), order='F')
        #idx = reshape(1:prod([ny,nx]),[ny,nx]);
        # local vertex numbering
        v1 = idx[:-1,:-1,:-1]
        v1 = v1.flatten('F')
        v1.shape = [v1.size, 1]
        
        v2 = idx[:-1,1:,:-1]
        v2 = v2.flatten('F')
        v2.shape = [v2.size, 1]
        
        v3 = idx[1:,:-1,:-1]
        v3 = v3.flatten('F')
        v3.shape = [v3.size, 1]
        
        v4 = idx[1:,1:,:-1]
        v4 = v4.flatten('F')
        v4.shape = [v4.size, 1]

        v5 = idx[:-1,:-1,1:]
        v5 = v5.flatten('F')
        v5.shape = [v5.size, 1]

        v6 = idx[:-1,1:,1:]
        v6 = v6.flatten('F')
        v6.shape = [v6.size, 1]

        v7 = idx[1:,:-1,1:]
        v7 = v7.flatten('F')
        v7.shape = [v7.size, 1]

        v8 = idx[1:,1:,1:]
        v8 = v8.flatten('F')
        v8.shape = [v8.size, 1]

        if UseBCC:
            # cell dimensions
            Cell_X = (1/N0)
            Cell_Y = (1/N1)
            Cell_Z = (1/N2)
            # apply scaling (translation already accounted for)
            Cell_X = L0*Cell_X
            Cell_Y = L1*Cell_Y
            Cell_Z = L2*Cell_Z

            # create BCC coordinates
            New_BCC = XP[v1[:,0],:]
            New_BCC[:,0] = New_BCC[:,0] + Cell_X/2
            New_BCC[:,1] = New_BCC[:,1] + Cell_Y/2
            New_BCC[:,2] = New_BCC[:,2] + Cell_Z/2

            Num_Cells = N0*N1*N2
            if (New_BCC.shape[0]!=Num_Cells):
                print("Number of cells does not match number of center vertices.")
                return

            v9 = np.arange(Num_Cells, dtype=VtxIndType) + XP.shape[0]
            v9.shape = [v9.size, 1]
            # v9 is linked to v1~v8
            XP = np.vstack((XP,New_BCC))

            # get v10
            v10 = np.zeros((Num_Cells, 1), dtype=VtxIndType)
            # recreate [TF, LOC] = ismember(v2,v1); % find the v2's that are also v1's
            TF = np.isin(v2[:,0], v1[:,0])
            v1_indices = np.arange(v1.shape[0], dtype=VtxIndType)
            dict_v1_and_indices = dict(zip(v1[:,0], v1_indices))
            LOC = list(map(lambda k: dict_v1_and_indices[k], v2[TF,0]))
            # note: v2(TF) == v1(LOC)
            v10[TF,0] = v9[LOC,0] # v10 corresponds to v9 at those
            # create X-shifted BCC coordinates
            New_BCC = XP[v2[~TF,0],:]
            New_BCC[:,0] = New_BCC[:,0] + Cell_X/2
            New_BCC[:,1] = New_BCC[:,1] + Cell_Y/2
            New_BCC[:,2] = New_BCC[:,2] + Cell_Z/2
            v10[~TF,0] = np.arange(New_BCC.shape[0], dtype=VtxIndType) + XP.shape[0]
            XP = np.vstack((XP,New_BCC)) # v10 is linked to v2

            # get v11
            v11 = np.zeros((Num_Cells, 1), dtype=VtxIndType)
            # recreate [TF, LOC] = ismember(v3,v1); % find the v3's that are also v1's
            TF = np.isin(v3[:,0], v1[:,0])
            LOC = list(map(lambda k: dict_v1_and_indices[k], v3[TF,0]))
            # note: v3(TF) == v1(LOC)
            v11[TF,0] = v9[LOC,0] # v11 corresponds to v9 at those
            # create Y-shifted BCC coordinates
            New_BCC = XP[v3[~TF,0],:]
            New_BCC[:,0] = New_BCC[:,0] + Cell_X/2
            New_BCC[:,1] = New_BCC[:,1] + Cell_Y/2
            New_BCC[:,2] = New_BCC[:,2] + Cell_Z/2
            v11[~TF,0] = np.arange(New_BCC.shape[0], dtype=VtxIndType) + XP.shape[0]
            XP = np.vstack((XP,New_BCC)) # v11 is linked to v3

            # get v12
            v12 = np.zeros((Num_Cells, 1), dtype=VtxIndType)
            # recreate [TF, LOC] = ismember(v5,v1); % find the v5's that are also v1's
            TF = np.isin(v5[:,0], v1[:,0])
            LOC = list(map(lambda k: dict_v1_and_indices[k], v5[TF,0]))
            # note: v5(TF) == v1(LOC)
            v12[TF,0] = v9[LOC,0] # v12 corresponds to v9 at those
            # create Z-shifted BCC coordinates
            New_BCC = XP[v5[~TF,0],:]
            New_BCC[:,0] = New_BCC[:,0] + Cell_X/2
            New_BCC[:,1] = New_BCC[:,1] + Cell_Y/2
            New_BCC[:,2] = New_BCC[:,2] + Cell_Z/2
            v12[~TF,0] = np.arange(New_BCC.shape[0], dtype=VtxIndType) + XP.shape[0]
            XP = np.vstack((XP,New_BCC)) # v12 is linked to v5

            # shift indices because we DO NOT use the first vertex!
            XP = XP[1:,:]
            del v1
            v2[:,0]  = v2[:,0] - 1
            v3[:,0]  = v3[:,0] - 1
            v4[:,0]  = v4[:,0] - 1
            v5[:,0]  = v5[:,0] - 1
            v6[:,0]  = v6[:,0] - 1
            v7[:,0]  = v7[:,0] - 1
            v8[:,0]  = v8[:,0] - 1
            v9[:,0]  = v9[:,0] - 1
            v10[:,0] = v10[:,0] - 1
            v11[:,0] = v11[:,0] - 1
            v12[:,0] = v12[:,0] - 1

            # should I have an option for only generating the vertices?
            
            if FlatSides:
                # project boundary vertex positions to the flat box sides

                # find all vertices above the x=Ptur[0], y=Ptur[1], and z=Ptur[2] planes
                x1_mask = XP[:,0] > Ptur[0] + 1E-11
                y1_mask = XP[:,1] > Ptur[1] + 1E-11
                z1_mask = XP[:,2] > Ptur[2] + 1E-11
                XP_ind = np.arange(XP.shape[0], dtype=VtxIndType)
                XP_x1_ind = XP_ind[x1_mask]
                XP_y1_ind = XP_ind[y1_mask]
                XP_z1_ind = XP_ind[z1_mask]

                # move those vertices to exactly the x=Ptur[0], y=Ptur[1], and z=Ptur[2] planes
                XP[XP_x1_ind,0] = Ptur[0]
                XP[XP_y1_ind,1] = Ptur[1]
                XP[XP_z1_ind,2] = Ptur[2]

                # copy those vertices, and move to exactly the x=Pbll[0], y=Pbll[1], and z=Pbll[2] planes
                XP_x0 = XP[XP_x1_ind,:]
                XP_x0[:,0] = Pbll[0]
                XP_y0 = XP[XP_y1_ind,:]
                XP_y0[:,1] = Pbll[1]
                XP_z0 = XP[XP_z1_ind,:]
                XP_z0[:,2] = Pbll[2]

                # add the vertex at the (mapped) origin
                XP_O = np.zeros((1,3), dtype=VtxIndType)
                XP_O[0,:] = Pbll[:]

                # collect the vertices (overwrite XP)
                XP = np.vstack((XP, XP_x0, XP_y0, XP_z0, XP_O))

                # now, mesh it
                D_tet = Delaunay(XP)
                TET = D_tet.simplices
                TET = TET.astype(VtxIndType)
                
            else:
                # create the cell connectivity
                
                # in the x-direction
                R0  = np.hstack((v9, v6, v2, v10))
                R1  = np.hstack((v9, v8, v6, v10))
                R2  = np.hstack((v9, v4, v8, v10))
                R3  = np.hstack((v9, v2, v4, v10))
                # in the y-direction
                R4  = np.hstack((v9, v8, v4, v11))
                R5  = np.hstack((v9, v7, v8, v11))
                R6  = np.hstack((v9, v3, v7, v11))
                R7  = np.hstack((v9, v4, v3, v11))
                # in the z-direction
                R8  = np.hstack((v9, v5, v6, v12))
                R9  = np.hstack((v9, v7, v5, v12))
                R10 = np.hstack((v9, v8, v7, v12))
                R11 = np.hstack((v9, v6, v8, v12))
                
                TET = np.vstack((R0,R1,R2,R3, R4,R5,R6,R7, R8,R9,R10,R11))

        else:
            # create the cell connectivity
            R0  = np.hstack((v1,v8,v3,v7))
            R1  = np.hstack((v1,v5,v8,v7))
            R2  = np.hstack((v1,v4,v3,v8))
            R3  = np.hstack((v1,v2,v4,v8))
            R4  = np.hstack((v1,v5,v6,v8))
            R5  = np.hstack((v1,v6,v2,v8))

            TET = np.vstack((R0,R1,R2,R3,R4,R5))

        # create the object
        VC = ahf_SM.VtxCoordType(3)
        VC.Set(XP)
        Mesh = ahf_SM.SimplexMesh(3,VC)
        Mesh.Set_Cell(TET)
        
        # finalize it!
        Mesh.Finalize_Mesh_Connectivity()
        Mesh.Close()

        return VC, Mesh

