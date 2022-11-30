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
        # computing only 1 affine map
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
        # computing M affine maps
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
        # converting only 1 point
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
        # converting M points
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

def Cartesian_To_Reference(vtx_coord, cart_coord):
    """Convert cartesian coordinates to reference element coordinates, where the reference
    element is the "standard" reference simplex.  Note: a projection is used to make
    sure the points are actually on the simplex.
    There are two ways to call this function:
    Inputs: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
            vertices of a single simplex of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            cart_coord: (GD,) numpy array that gives the cartesian coordinates of
            the given point.
    Output: (TD,) numpy array that gives the coordinates of the point in the
            reference simplex.
    OR
    Inputs: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
            vertices of M simplices of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            cart_coord: (M,GD) numpy array that gives the cartesian coordinates of
            the given M points.
    Output: (M,TD) numpy array that gives the coordinates of the M points in the
            reference simplex.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return
    if type(cart_coord) is not np.ndarray:
        print("Error: cart_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # converting only 1 point
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        dim_cc = cart_coord.shape
        if dim_cc[0]!=GD:
            print("Error: cart_coord must be a numpy array of shape (GD,) if vtx_coord is (TD+1,GD)!")
            return

        # get the affine map for the given simplex
        A, b = Affine_Map(vtx_coord)

        # compute A' * (xc - b)
        xd = np.zeros((GD,), dtype=CoordType)
        xd[:] = cart_coord[:] - b[:,0]
        r = np.matmul(A.T, xd)
        # compute M = A'*A
        M = np.matmul(A.T, A)
        # solve for reference coordinates
        ref_coord = np.linalg.solve(M, r)

    else:
        # converting M points
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        dim_cc = cart_coord.shape
        if (dim_cc[0]!=M) or (dim_cc[1]!=GD):
            print("Error: cart_coord must be a numpy array of shape (M,GD) if vtx_coord is (M,TD+1,GD)!")
            return

        # get the affine maps for the given simplices
        A, b = Affine_Map(vtx_coord)

        # compute A' * (xc - b)
        xd = np.zeros((M,GD,1), dtype=CoordType)
        xd[:,:,0] = cart_coord[:,:] - b[:,:,0]
        A_T = np.transpose(A, (0, 2, 1))
        r = np.matmul(A_T, xd[:,:,[0]])
        
        # compute M = A'*A
        M = np.matmul(A_T, A)
        # solve for reference coordinates
        ref_coord = np.linalg.solve(M, r)
        # leave off extraneous dimension
        ref_coord = ref_coord[:,:,0]

    return ref_coord

def Barycentric_To_Reference(bary_coord):
    """Convert barycentric coordinates to reference element coordinates, where the reference
    element is the "standard" reference simplex.
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
    if type(bary_coord) is not np.ndarray:
        print("Error: bary_coord must be a numpy array!")
        return

    ndim   = bary_coord.ndim
    dim_bc = bary_coord.shape
    if ndim==1:
        # converting only 1 point
        TD = dim_bc[0]-1
        ref_coord = bary_coord[1:]
        ref_coord = np.reshape(ref_coord, (TD,))
    else:
        # converting M points
        M  = dim_bc[0]
        TD = dim_bc[1]-1
        
        ref_coord = bary_coord[:,1:]
        ref_coord = np.reshape(ref_coord, (M,TD))

    return ref_coord

def Reference_To_Barycentric(ref_coord):
    """Convert reference element coordinates to barycentric coordinates, where the reference
    element is the "standard" reference simplex.
    There are two ways to call this function:
    Input: ref_coord: (TD,) numpy array that gives the coordinates of the point in the
           reference simplex.
    Output: (TD+1,) numpy array that gives the barycentric coordinates of the given point.
    OR
    Input: ref_coord: (M,TD) numpy array that gives the coordinates of the M points in the
           reference simplex.
    Output: (M,TD+1) numpy array that gives the barycentric coordinates of the given M points.
    """
    if type(ref_coord) is not np.ndarray:
        print("Error: ref_coord must be a numpy array!")
        return

    ndim   = ref_coord.ndim
    dim_rc = ref_coord.shape
    if ndim==1:
        # converting only 1 point
        TD = dim_rc[0]
        bary_coord = np.zeros((TD+1,), dtype=CoordType)
        
        bary_coord[1:] = ref_coord[:]
        bary_coord[0]  = 1.0 - np.sum(ref_coord)
    else:
        # converting M points
        M  = dim_rc[0]
        TD = dim_rc[1]
        bary_coord = np.zeros((M,TD+1), dtype=CoordType)
        
        bary_coord[:,1:] = ref_coord[:,:]
        bary_coord[:,0]  = 1.0 - np.sum(ref_coord, axis=1)

    return bary_coord

def Barycentric_To_Cartesian(vtx_coord, bary_coord):
    """Convert barycentric (simplex) coordinates to cartesian coordinates.
    There are two ways to call this function:
    Inputs: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
            vertices of a single simplex of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            bary_coord: a (TD+1,) numpy array that gives the barycentric
            coordinates of the point w.r.t. the given simplex coordinates.
    Output: (GD,) numpy array that gives the cartesian coordinates of the given point.
    OR
    Inputs: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
            vertices of M simplices of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            bary_coord: a (M,TD+1) numpy array that gives the barycentric
            coordinates of M points w.r.t. the corresponding simplex coordinates.
    Output: (M,GD) numpy array that gives the cartesian coordinates of the given M points.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return
    if type(bary_coord) is not np.ndarray:
        print("Error: bary_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # converting only 1 point
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        dim_bc = bary_coord.shape
        if dim_bc[0]!=TD+1:
            print("Error: bary_coord must be a numpy array of shape (TD+1,) if vtx_coord is (TD+1,GD)!")
            return

        cart_coord = np.zeros((GD,), dtype=CoordType)
        for tt in np.arange(0, (TD+1), dtype=SmallIndType):
            # add contributions
            cart_coord[:] += bary_coord[tt] * vtx_coord[tt,:]

    else:
        # converting M points
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        dim_bc = bary_coord.shape
        if (dim_bc[0]!=M) or (dim_bc[1]!=TD+1):
            print("Error: bary_coord must be a numpy array of shape (M,TD+1) if vtx_coord is (M,TD+1,GD)!")
            return

        cart_coord = np.zeros((M,GD), dtype=CoordType)
        for tt in np.arange(0, (TD+1), dtype=SmallIndType):
            # add contributions
            cart_coord[:,:] += vtx_coord[:,tt,:] * bary_coord[:, [tt]]

    return cart_coord

def Cartesian_To_Barycentric(vtx_coord, cart_coord):
    """Convert cartesian coordinates to barycentric (simplex) coordinates.
    Note: a projection is used to make sure the points are actually on the simplex.
    There are two ways to call this function:
    Inputs: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
            vertices of a single simplex of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            cart_coord: (GD,) numpy array that gives the cartesian coordinates of
            the given point.
    Output: (TD+1,) numpy array that gives the barycentric coordinates of the given point.
    OR
    Inputs: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
            vertices of M simplices of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            cart_coord: (M,GD) numpy array that gives the cartesian coordinates of
            the given M points.
    Output: (M,TD+1) numpy array that gives the barycentric coordinates of the given M points.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return
    if type(cart_coord) is not np.ndarray:
        print("Error: cart_coord must be a numpy array!")
        return

    # first, convert from cartesian to reference domain coordinates
    ref_coord  = Cartesian_To_Reference(vtx_coord, cart_coord)
    bary_coord = Reference_To_Barycentric(ref_coord)

    return bary_coord

def Diameter(vtx_coord):
    """Compute the diameter of a simplex.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: a number that gives the diameter of the simplex (max edge length).
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: (M,) numpy array that gives the diameters of the given M simplices.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing diameter for one simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        # compute all edge lengths, and take the MAX
        MAX_Edge_Length = 0.0
        for rr in np.arange(0, (TD+1), dtype=SmallIndType):
            for cc in np.arange(rr+1, (TD+1), dtype=SmallIndType):
                diff_xc = vtx_coord[rr,:] - vtx_coord[cc,:]
                L0 = np.sqrt(np.sum(diff_xc**2))
                if (L0 > MAX_Edge_Length):
                    MAX_Edge_Length = L0
    else:
        # computing diameter for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]

        # compute all edge lengths, and take the MAX
        MAX_Edge_Length = np.zeros((M,), dtype=RealType)
        for rr in np.arange(0, (TD+1), dtype=SmallIndType):
            for cc in np.arange(rr+1, (TD+1), dtype=SmallIndType):
                diff_xc = vtx_coord[:,rr,:] - vtx_coord[:,cc,:]
                L0 = np.sqrt(np.sum(diff_xc**2, axis=1))
                MAX_Edge_Length = np.where(L0 < MAX_Edge_Length, MAX_Edge_Length, L0)

    return MAX_Edge_Length

def Bounding_Box(vtx_coord):
    """Compute the bounding (cartesian) box of a simplex.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: BB_min, BB_max.  Both are (GD,) numpy arrays that contain the minimum
             and maximum coordinates of the "box".
       Example:  if GD==3, then
             BB_min[:] = [X_min, Y_min, Z_min], (numpy array),
             BB_max[:] = [X_max, Y_max, Z_max], (numpy array).
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: BB_min, BB_max.  Both are (M,GD) numpy arrays that contain the minimum
             and maximum coordinates of the "box".
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing bounding for one simplex
        # TD = dim_vc[0]-1
        # GD = dim_vc[1]
        
        # compute the box
        BB_min = np.amin(vtx_coord, axis=0)
        BB_max = np.amax(vtx_coord, axis=0)

    else:
        # computing diameter for M simplices
        # M  = dim_vc[0]
        # TD = dim_vc[1]-1
        # GD = dim_vc[2]

        # compute the box
        BB_min = np.amin(vtx_coord, axis=1)
        BB_max = np.amax(vtx_coord, axis=1)

    return BB_min, BB_max

def Volume(vtx_coord):
    """Compute the volume of a simplex.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: a number that gives the (TD)-dimensional volume of the simplex.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: (M,) numpy array that gives the (TD)-dimensional volumes of the M simplices.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing volume for one simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        # get the affine map for the simplex
        A, b = Affine_Map(vtx_coord)

        if (TD==0):
            # nothing to do!
            det_A = 0.0
        elif (GD==TD):
            det_A = np.linalg.det(A)
        else:
            # must use the metric, compute G = A'*A
            G = np.matmul(A.T, A)
            det_G = np.linalg.det(G)
            det_A = np.sqrt(det_G)
    else:
        # computing volume for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]

        # get the affine maps for the simplices
        A, b = Affine_Map(vtx_coord)

        if (TD==0):
            # nothing to do!
            det_A = np.zeros((M,), dtype=RealType)
        elif (GD==TD):
            det_A = np.linalg.det(A)
        else:
            # must use the metric, compute G = A'*A
            A_T = np.transpose(A, (0, 2, 1))
            # compute G = A'*A
            G = np.matmul(A_T, A)
            det_G = np.linalg.det(G)
            det_A = np.sqrt(det_G)

    Fact0 = np.math.factorial(TD)
    Vol = (1.0/Fact0) * det_A
    return Vol

def Perimeter(vtx_coord):
    """Compute the perimeter of a simplex.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: Perimeter: a number giving the total (TD-1)-dimensional volume
             ("surface area") of the boundary of the simplex;
             Facet_Vol: (TD+1,) numpy array that gives the (TD-1)-dimensional
             volume of each of the TD+1 facets of the simplex.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: Perimeter: (M,) numpy array giving the total (TD-1)-dimensional volume
             ("surface area") of the boundary of the M simplices;
             Facet_Vol: (M,TD+1) numpy array that gives the (TD-1)-dimensional volume
             of each of the TD+1 facets of the M simplices.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing perimeter for one simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        Facet_Vol = np.zeros((TD+1,), dtype=RealType)
        if (TD==1):
            # counting measure
            Facet_Vol[0] = 1.0
            Facet_Vol[1] = 1.0
        else:
            # loop through each facet of the simplex
            All_VI = np.arange(0, (TD+1), dtype=SmallIndType)
            for ff in All_VI:
                # get the vertex indices of the current facet
                Facet_VI = np.delete(All_VI, ff)
                Facet_vc = vtx_coord[Facet_VI,:]
                # compute it's volume (or "surface area")
                Facet_Vol[ff] = Volume(Facet_vc)

        Perimeter = np.sum(Facet_Vol)
    else:
        # computing diameter for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]

        Facet_Vol = np.zeros((M,TD+1), dtype=RealType)
        if (TD==1):
            # counting measure
            Facet_Vol[:,0] = 1.0
            Facet_Vol[:,1] = 1.0
        else:
            # loop through each facet of the simplex
            All_VI = np.arange(0, (TD+1), dtype=SmallIndType)
            for ff in All_VI:
                # get the vertex indices of the current facet
                Facet_VI = np.delete(All_VI, ff)
                Facet_vc = vtx_coord[:,Facet_VI,:]
                # compute it's volume (or "surface area")
                Facet_Vol[:,ff] = Volume(Facet_vc)

        Perimeter = np.sum(Facet_Vol, axis=1)

    return Perimeter, Facet_Vol

def Orthogonal_Frame(vtx_coord,frame_type="all"):
    """Get orthonormal column vectors that describe a frame for a set of given
    simplices.  If the keyword "all" is given, then the orthonormal vectors
    span all of \R^{GD}, where the first TD vectors span the *tangent* space
    of the simplex, and the last (GD - TD) vectors span the space *orthogonal*
    to the tangent space (i.e. the *normal* space).  The keyword "tangent" only
    gives the vectors in the tangent space; the keyword "normal" only gives the
    vectors in the normal space.

    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD;
           frame_type: a string = "all", "tangent", or "normal".  If omitted,
           then default it "all".  "all" gives a complete frame of \R^{GD},
           "tangent" gives the tangent space, and "normal" gives the normal space.
    Outputs: Ortho: numpy matrix (GD,GD), whose column vectors are the (unit)
             basis vectors of the frame.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD;
           frame_type: see above.
    Outputs: Ortho: numpy array (M,GD,GD), which is a stack of M matrices, each
             of whose column vectors are the (unit) basis vectors of the frame.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    if frame_type.lower() == "tangent":
        qr_type = "reduced"
    else:
        qr_type = "complete"

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing only 1 frame
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        # get the affine map for the given simplex
        A, b = Affine_Map(vtx_coord)
        #A_rank = np.linalg.matrix_rank(A)
        
        # QR it!
        Ortho, R = np.linalg.qr(A, mode=qr_type)
        if qr_type == "complete":
            # make it have positive determinant
            O_det = np.linalg.det(Ortho)
            if (O_det < 0.0):
                Ortho[:,0] = -Ortho[:,0]
            # adjust for orientation when GD==TD+1, i.e. keep the orientation consistent with A.
            if (GD==TD+1):
                # append the normal vector to the Jacobian matrix
                Test_Mat = np.hstack((A, Ortho[:,[-1]]))
                Test_det = np.linalg.det(Test_Mat)
                if (Test_det < 0.0):
                    # flip the orientation of the normal vector...
                    Ortho[:,-1] = -Ortho[:,-1]
                    # ... and the tangent space (because we already flipped sign of determinant above)
                    Ortho[:,0] = -Ortho[:,0]

        if frame_type.lower() == "normal":
            # only give the last column vectors
            Ortho = Ortho[:,TD:GD]
    else:
        # computing M frames
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        # get the affine maps for the given simplices
        A, b = Affine_Map(vtx_coord)
        #A_rank = np.linalg.matrix_rank(A)
        
        # QR it!
        Ortho, R = np.linalg.qr(A, mode=qr_type)
        if qr_type == "complete":
            # make it have positive determinant
            O_det = np.linalg.det(Ortho)
            
            sign_det = np.sign(O_det)
            sign_det = sign_det.reshape((M,1))
            sign_det_rep = np.tile(sign_det, (1, GD))
            Ortho[:,:,0] = sign_det_rep[:,:] * Ortho[:,:,0]

            # adjust for orientation when GD==TD+1, i.e. keep the orientation consistent with A.
            if (GD==TD+1):
                # append the normal vector to the Jacobian matrix
                Test_Mat = 0*Ortho
                Test_Mat[:,:,0:GD-1] = A[:,:,0:GD-1]
                Test_Mat[:,:,-1] = Ortho[:,:,-1]
                Test_det = np.linalg.det(Test_Mat)

                # adjust based on the sign
                sign_det = np.sign(Test_det)
                sign_det = sign_det.reshape((M,1))
                sign_det_rep = np.tile(sign_det, (1, GD))
                # flip the orientation of the normal vector...
                Ortho[:,:,-1] = sign_det_rep[:,:] * Ortho[:,:,-1]
                # ... and the tangent space (because we already flipped sign of determinant above)
                Ortho[:,:,0] = sign_det_rep[:,:] * Ortho[:,:,0]

        if frame_type.lower() == "normal":
            # only give the last column vectors
            Ortho = Ortho[:,:,TD:GD]

    return Ortho

def Tangent_Space(vtx_coord):
    """Get orthonormal column vectors that describe the tangent space of a
    set of given simplices.  The number of tangent vectors is TD, and they
    live in \R^{GD}.

    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: TS: numpy matrix (GD,TD), whose column vectors are the (unit)
             basis vectors of the tangent space.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: TS: numpy array (M,GD,TD), which is a stack of M matrices, each of
             whose column vectors are the (unit) basis vectors of the tangent space.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing only 1 tangent space
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        if (TD==0):
            # do nothing; tangent space is empty
            TS = np.zeros((GD,0), dtype=RealType)
        elif (TD==1):
            # special case: 1-D space curve
            A, b = Affine_Map(vtx_coord)
            
            # normalize the one column vector of A to get the tangent vector
            TS = A * (1.0 / np.linalg.norm(A,2))
        else:
            # the general case
            TS = Orthogonal_Frame(vtx_coord,frame_type="tangent")
    else:
        # computing M tangent spaces
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        if (TD==0):
            # do nothing; tangent space is empty
            TS = np.zeros((M,GD,0), dtype=RealType)
        elif (TD==1):
            # special case: 1-D space curve
            A, b = Affine_Map(vtx_coord)
            
            # normalize the one column vector of A to get the tangent vector
            A_norm = np.linalg.norm(A[:,:,0], ord=2, axis=1)
            A_norm_rep = np.tile(A_norm, (1, GD))
            TS = A[:,:,[0]] * (1.0 / A_norm_rep[:,:])
        else:
            # the general case
            TS = Orthogonal_Frame(vtx_coord,frame_type="tangent")

    return TS

def Normal_Space(vtx_coord):
    """Get orthonormal column vectors that describe the space *orthogonal*
    to the tangent space of a set of given simplices (i.e. the normal space).
    The number of normal vectors is GD-TD, and they live in \R^{GD}.

    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: NS: numpy matrix (GD,GD-TD), whose column vectors are the (unit)
             basis vectors of the normal space.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: NS: numpy array (M,GD,GD-TD), which is a stack of M matrices, each of
             whose column vectors are the (unit) basis vectors of the normal space.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing only 1 normal space
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        if (TD==0):
            # its all of \R^{GD}
            NS = np.identity(GD, dtype=RealType)
        elif (TD==GD):
            # do nothing; normal space is empty
            NS = np.zeros((GD,0), dtype=RealType)
        elif ( (TD==1) and (GD==2) ):
            # special case: curve in \R^2
            A, b = Affine_Map(vtx_coord)
            
            # rotate the column vector from A
            V = np.array([ [-A[1,0]], [A[0,0]] ], dtype=RealType)
            # normalize to get the normal vector
            NS = V * (1.0 / np.linalg.norm(V,2))
        elif ( (TD==2) and (GD==3) ):
            # special case: surface in \R^3
            A, b = Affine_Map(vtx_coord)
            
            # take cross product of the two column vectors of A
            V = np.zeros((GD,1), dtype=RealType)
            # compute cross-product
            V[0,0] =   A[1,0] * A[2,1] - A[2,0] * A[1,1]
            V[1,0] = -(A[0,0] * A[2,1] - A[2,0] * A[0,1])
            V[2,0] =   A[0,0] * A[1,1] - A[1,0] * A[0,1]
            
            # normalize to get the normal vector
            NS = V * (1.0 / np.linalg.norm(V,2))
        else:
            # general case
            NS = Orthogonal_Frame(vtx_coord,frame_type="normal")
    else:
        # computing M normal spaces
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        if (TD==0):
            # its all of \R^{GD}
            I0 = np.identity(GD, dtype=RealType)
            NS = np.tile(I0, (M,1,1))
        elif (TD==GD):
            # do nothing; normal space is empty
            NS = np.zeros((M,GD,0), dtype=RealType)
        elif ( (TD==1) and (GD==2) ):
            # special case: curve in \R^2
            A, b = Affine_Map(vtx_coord)
            
            # rotate the column vector from A
            NS = np.zeros((M,GD,1), dtype=RealType)
            NS[:,0,0] = -A[:,1,0]
            NS[:,1,0] =  A[:,0,0]
            # normalize to get the normal vector
            NS_norm = np.linalg.norm(NS[:,:,[0]], ord=2, axis=1)
            NS_norm_rep = np.tile(NS_norm, (1, GD))
            NS[:,:,0] = NS[:,:,0] * (1.0 / NS_norm_rep[:,:])
        elif ( (TD==2) and (GD==3) ):
            # special case: surface in \R^3
            A, b = Affine_Map(vtx_coord)
            
            # take cross product of the two column vectors of A
            NS = np.zeros((M,GD,1), dtype=RealType)
            # compute cross-product
            NS[:,0,0] =   A[:,1,0] * A[:,2,1] - A[:,2,0] * A[:,1,1]
            NS[:,1,0] = -(A[:,0,0] * A[:,2,1] - A[:,2,0] * A[:,0,1])
            NS[:,2,0] =   A[:,0,0] * A[:,1,1] - A[:,1,0] * A[:,0,1]
            # normalize to get the normal vector
            NS_norm = np.linalg.norm(NS[:,:,[0]], ord=2, axis=1)
            NS_norm_rep = np.tile(NS_norm, (1, GD))
            NS[:,:,0] = NS[:,:,0] * (1.0 / NS_norm_rep[:,:])
        else:
            # general case
            NS = Orthogonal_Frame(vtx_coord,frame_type="normal")

    return NS

def Hyperplane_Closest_Point(vtx_coord,pY):
    """Find the closest point, X_star, on a hyperplane to a given point, Y.
    The hyperplane is spanned by the tangent space of a given simplex.
    This also returns the difference vector:  NV = Y - X_star.
    Note: the closest point, X_star, may not lie inside the simplex.

    There are two ways to call this function:
    Inputs: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
            vertices of a single simplex of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            pY: a (GD,1) numpy array that gives the coordinates of the given
            point Y in the ambient space.
    Outputs: X_star: a (GD,1) numpy array representing the coordinates of the
             closest point;
             NV: a (GD,1) numpy array representing the components of NV.
    OR
    Inputs: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
            vertices of M simplices of topological dimension TD embedded in
            a Euclidean space of dimension GD;
            pY: a (M,GD,1) numpy array that gives the coordinates of M given
            points Y in the ambient space.
    Outputs: X_star: a (M,GD,1) numpy array representing the coordinates of the
             M closest points;
             NV: a (M,GD,1) numpy array representing the components of the M
             difference vectors, NV.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return
    if type(pY) is not np.ndarray:
        print("Error: pY must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing for one simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]
        # compute the co-dimension, i.e. the dimension of the normal space
        ND = GD-TD

        if (ND==0):
            # easy case: X_star = Y
            X_star = np.zeros((GD,1), dtype=CoordType)
            X_star[:,0] = pY[:,0]
            NV     = np.zeros((GD,1), dtype=RealType)
        elif (ND > TD):
            # normal space is too big, so use the tangent space
            TS = Tangent_Space(vtx_coord)
            
            # compute the difference vector between Y and
            #    one of the points of the simplex
            DIFF = np.zeros((GD,1), dtype=RealType)
            DIFF[:,0] = pY[:,0] - vtx_coord[0,:]
            
            # project onto the tangent space
            TV = np.zeros((GD,1), dtype=RealType)
            for jj in range(TD):
                TV[:,0] += np.dot(DIFF[:,0],TS[:,jj]) * TS[:,jj]

            # now compute the normal vector
            X_star = np.zeros((GD,1), dtype=CoordType)
            X_star[:,0] = vtx_coord[0,:] + TV[:,0]
            NV = pY - X_star
        else:
            # tangent space is too big, so use the normal space
            NS = Normal_Space(vtx_coord)

            # compute the difference vector between Y and
            #    one of the points of the simplex
            DIFF = np.zeros((GD,1), dtype=RealType)
            DIFF[:,0] = pY[:,0] - vtx_coord[0,:]

            # project onto the normal space
            NV = np.zeros((GD,1), dtype=RealType)
            for jj in range(ND):
                NV[:,0] += np.dot(DIFF[:,0],NS[:,jj]) * NS[:,jj]
            # compute X_star
            X_star = pY - NV
    else:
        # computing for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        # compute the co-dimension, i.e. the dimension of the normal space
        ND = GD-TD

        if (ND==0):
            # easy case: X_star = Y
            X_star = np.zeros((M,GD,1), dtype=CoordType)
            X_star[:,:,0] = pY[:,:,0]
            NV = np.zeros((M,GD,1), dtype=RealType)
        elif (ND > TD):
            # normal space is too big, so use the tangent space
            TS = Tangent_Space(vtx_coord)
            
            # compute the difference vector between Y and
            #    one of the points of the simplex
            DIFF = np.zeros((M,GD,1), dtype=RealType)
            DIFF[:,:,0] = pY[:,:,0] - vtx_coord[:,0,:]
            
            # project onto the tangent space
            TV = np.zeros((M,GD,1), dtype=RealType)
            for jj in range(TD):
                dot_prod = np.sum(TS[:,:,jj] * DIFF[:,:,0], axis=1)
                dot_prod = dot_prod.reshape((M,1))
                dot_prod_rep = np.tile(dot_prod, (1, GD))
                TV[:,:,0] += dot_prod_rep[:,:] * TS[:,:,jj]

            # now compute the normal vector
            X_star = np.zeros((M,GD,1), dtype=CoordType)
            X_star[:,:,0] = vtx_coord[:,0,:] + TV[:,:,0]
            NV = pY - X_star
        else:
            # tangent space is too big, so use the normal space
            NS = Normal_Space(vtx_coord)

            # compute the difference vector between Y and
            #    one of the points of the simplex
            DIFF = np.zeros((M,GD,1), dtype=RealType)
            DIFF[:,:,0] = pY[:,:,0] - vtx_coord[:,0,:]

            # project onto the normal space
            NV = np.zeros((M,GD,1), dtype=RealType)
            for jj in range(ND):
                dot_prod = np.sum(NS[:,:,jj] * DIFF[:,:,0], axis=1)
                dot_prod = dot_prod.reshape((M,1))
                dot_prod_rep = np.tile(dot_prod, (1, GD))
                NV[:,:,0] += dot_prod_rep[:,:] * NS[:,:,jj]
            # compute X_star
            X_star = pY - NV

    return X_star, NV

def Barycenter(vtx_coord):
    """Compute the barycenter of a simplex.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: (GD,) numpy array that gives the cartesian coordinates of the barycenter
            of the simplex.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Output: (M,GD) numpy array that gives the cartesian coordinates of the barycenters
            of the given M simplices.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing for only 1 simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        bary_coord = (1.0/(TD+1)) * np.ones((TD+1,), dtype=CoordType)
        cart_coord = Barycentric_To_Cartesian(vtx_coord, bary_coord)

    else:
        # computing for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        bary_coord = (1.0/(TD+1)) * np.ones((M,TD+1), dtype=CoordType)
        cart_coord = Barycentric_To_Cartesian(vtx_coord, bary_coord)
        
    return cart_coord

def Circumcenter(vtx_coord):
    """Compute the circumcenter and circumradius of a simplex.
    see:  https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm#Coxeter
    for the method.
    There are two ways to call this function:
    Input: vtx_coord: a (TD+1,GD) numpy array that gives the coordinates of the
           vertices of a single simplex of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: CB: (TD+1,) numpy array that gives the barycentric coordinates of the
             circumcenter of the simplex;
             CR: a single number that gives the circumradius.
    OR
    Input: vtx_coord: a (M,TD+1,GD) numpy array that gives the coordinates of the
           vertices of M simplices of topological dimension TD embedded in
           a Euclidean space of dimension GD.
    Outputs: CB: (M,TD+1) numpy array that gives the barycentric coordinates of the
             circumcenters of the given M simplices;
             CR: (M,) numpy array that gives the corresponding circumradii.
    """
    if type(vtx_coord) is not np.ndarray:
        print("Error: vtx_coord must be a numpy array!")
        return

    ndim   = vtx_coord.ndim
    dim_vc = vtx_coord.shape
    if ndim==1:
        print("Error: vtx_coord must be a numpy array of shape (TD+1,GD) or (M,TD+1,GD)!")
        return
    elif ndim==2:
        # computing for only 1 simplex
        TD = dim_vc[0]-1
        GD = dim_vc[1]

        # allocate (Cayley-Menger) matrix
        MM = np.zeros((TD+2,TD+2), dtype=RealType)

        # set the first row (first entry is zero)
        MM[0,1:] = 1.0

        # now set the rest, based on length of simplex edges (squared)
        for rr in range(TD+1):
            # get the (rr)-th vertex
            P_rr = vtx_coord[rr,:]
            for cc in range(rr+1,TD+1):
                # get the (cc)-th vertex
                P_cc = vtx_coord[cc,:]
                # take the difference
                DD = P_rr - P_cc
                # length squared
                MM[rr+1,cc+1] = np.sum(DD**2)

        # now set the lower triangular part (it is a symmetric matrix)
        for rr in range(TD+2):
            for cc in range(rr+1,TD+2):
                MM[cc,rr] = MM[rr,cc]

        # compute the inverse
        MM_inv = np.linalg.inv(MM)
        # error check
        if (MM_inv[0,0] > 0.0):
            print("Error: Fatal error in 'Simplex_Circumcenter'!")
            print("       The circumradius was invalid!")
            return

        # output barycentric coordinates
        CB = np.zeros((TD+1,), dtype=CoordType)
        CB[0:] = MM_inv[0,1:]
        # output circumradius
        CR = np.sqrt(-MM_inv[0,0] / 2.0)

    else:
        # computing for M simplices
        M  = dim_vc[0]
        TD = dim_vc[1]-1
        GD = dim_vc[2]
        
        # allocate M (Cayley-Menger) matrices
        MM = np.zeros((M,TD+2,TD+2), dtype=RealType)

        # set the first row (first entry is zero)
        MM[:,0,1:] = 1.0

        # now set the rest, based on length of simplex edges (squared)
        for rr in range(TD+1):
            # get the (rr)-th vertex
            P_rr = vtx_coord[:,rr,:]
            for cc in range(rr+1,TD+1):
                # get the (cc)-th vertex
                P_cc = vtx_coord[:,cc,:]
                # take the difference
                DD = P_rr - P_cc
                # length squared
                len_sq = np.sum(DD**2, axis=1)
                MM[:,rr+1,cc+1] = len_sq[:]

        # now set the lower triangular part (it is a symmetric matrix)
        for rr in range(TD+2):
            for cc in range(rr+1,TD+2):
                MM[:,cc,rr] = MM[:,rr,cc]

        # compute the inverse
        MM_inv = np.linalg.inv(MM)
        # error check
        if (np.amax(MM_inv[:,0,0]) > 0.0):
            print("Error: Fatal error in 'Simplex_Circumcenter'!")
            print("       The circumradius was invalid!")
            return

        # output barycentric coordinates
        CB = np.zeros((M,TD+1), dtype=CoordType)
        CB[:,0:] = MM_inv[:,0,1:]
        # output circumradius
        CR = np.zeros((M,), dtype=CoordType)
        CR[:] = np.sqrt(-MM_inv[:,0,0] / 2.0)
        
    return CB, CR



# next center...


