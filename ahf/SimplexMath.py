"""
ahf.SimplexMath.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Algorithms for computing with simplices. Contains mathematical sub-routines,
depending on the topological dimension TD, and/or the geometric dimension GD.

Also, see "SimplexMesh.py" for more explanation.

Copyright (c) 09-04-2022,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *


def Affine_Map(vtx_coord):
    """Get the (jacobian) matrix and translation vector for the affine map from the
    'standard' reference simplex. This is the usual finite element affine map.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: A is the jacobian matrix of shape (GD,TD);
             b is the translation vector of shape (GD,1).
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: A contains M jacobian matrices, with shape (M,GD,TD);
             b contains M translation vectors of shape (M,GD,1).
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: input must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: input must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        # translation vector
        b = np.zeros((GD,1), dtype=CoordType)
        b[:] = (vtx_coord[[0],:]).T
        
        # Jacobian matrix
        A = np.zeros((GD,TD), dtype=CoordType)
        for tt in np.arange(1, TD+1, dtype=SmallIndType):
            # special case of a linear simplex:
            A[:,[tt-1]] = (vtx_coord[[tt],:]).T - b
    else:
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]

        # translation vectors
        b = np.zeros((M,GD,1), dtype=CoordType)
        b[:,:,0] = vtx_coord[:,0,:]

        # Jacobian matrices
        A = np.zeros((M,GD,TD), dtype=CoordType)
        for tt in np.arange(1, TD+1, dtype=SmallIndType):
            # special case of a linear simplex:
            A[:,:,(tt-1)] = vtx_coord[:,tt,:] - b[:,:,0]

    return A, b

def Reference_To_Cartesian(vtx_coord, ref_coord):
    """Convert reference element coordinates to cartesian coordinates, where the reference
    element is the "standard" reference simplex.
    There are two ways to call this function:
    Inputs: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
            vertices of a single simplex of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            ref_coord: a (TD,) numpy array that gives the coordinates of a point in the
            reference simplex.
    Output: (GD,) numpy array that gives the cartesian coordinates of the given point.
    OR
    Inputs: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
            vertices of M simplices of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            ref_coord: a (M,TD) numpy array that gives the coordinates of M points in the
            reference simplex.
    Output: (M,GD) numpy array that gives the cartesian coordinates of the given M points.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return
    if type(ref_coord) is not np.ndarray:
        print("Error: ref_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        dim_rc = ref_coord.shape
        if dim_rc[0]!=TD:
            print("Error: ref_coord must be a numpy array of shape (TD,) if vtx_coord is (TD+1,GD)!")
            return

        # note: this is computed using (effectively) barycentric coordinates

        # compute 0th barycentric coordinate
        BC0 = 1.0 - np.sum(ref_coord)

        # init to 0th contribution
        cart_coord = np.zeros((GD,), dtype=CoordType)
        cart_coord[:] = BC0 * vtx_coord[0,:]
        
        for tt in np.arange(1, (TD+1), dtype=SmallIndType):
            # add contribution (from other barycentric coordinates)
            cart_coord[:] += ref_coord[tt-1] * vtx_coord[tt,:]

    else:
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        dim_rc = ref_coord.shape
        if (dim_rc[0]!=M) or (dim_rc[1]!=TD):
            print("Error: ref_coord must be a numpy array of shape (M,TD) if vtx_coord is (M,TD+1,GD)!")
            return

        # note: this is computed using (effectively) barycentric coordinates

        # compute 0th barycentric coordinate
        BC0 = 1.0 - np.sum(ref_coord, axis=1)

        # init to 0th contribution
        cart_coord = np.zeros((M,GD), dtype=CoordType)
        cart_coord[:,:] = vtx_coord[:,0,:] * BC0[:, np.newaxis]
        
        for tt in np.arange(1, (TD+1), dtype=SmallIndType):
            # add contribution (from other barycentric coordinates)
            cart_coord[:,:] += vtx_coord[:,tt,:] * ref_coord[:, [tt-1]]

    return cart_coord




