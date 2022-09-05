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
    """Get the (jacobian) matrix and translation vector for the affine map
    from the 'standard' reference simplex. This is the usual finite element
    affine map.
    Input: vtx_coord is a (TD+1,GD) numpy array that gives the coordinates of
    the vertices of a simplex of topological dimension TD embedded in a
    Euclidean space of dimension GD.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: input must be a numpy array!")
        return

    dim_vc = vtx_coord.shape
    if len(dim_vc)==1:
        print("Error: input must be a numpy array with a 2-D shape!")
        return
    else:
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

    return A, b

