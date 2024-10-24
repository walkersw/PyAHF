"""
ahf.fem.Formulation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Definition of simple classes for defining a finite element formulation.

Note: this is experimental.

Copyright (c) 10-18-2024,  Shawn W. Walker
"""

import numpy as np
import scipy as sp
import math
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, CoordType
from ahf import *

import ahf.SimplexMath as sm
from ahf.SimplexMesh import *

# define data types

# # a single mesh edge (two connected vertices)
# # An edge is defined by [v0, v1], where v0, v1 are the end vertices of the edge.
# # For a simplex mesh, the edge [v0, v1] exists when v0, v1 are both
# # contained in the *same* mesh cell.
# MeshEdgeType = np.dtype({'names': ['v0', 'v1'],
                         # 'formats': [VtxIndType, VtxIndType],
                         # 'titles': ['global tail index', 'global head index']})
# NULL_MeshEdge = np.array((NULL_Vtx, NULL_Vtx), dtype=MeshEdgeType)


class FEspace:
    """
    Class for setting up a finite element space.

    Note: ???
    """

    def __init__(self, name=None, mesh=None):
        """
        ?????? CELL_DIM >= 0 is the topological dimension that the cells live in.
        res_buf in [0.0, 1.0] is optional, and is for reserving extra space for cell data.
        """
        # if (CELL_DIM<0):
            # print("Error: cell dimension must be non-negative!")
        # assert(CELL_DIM>=0)
        # if np.rint(CELL_DIM).astype(SmallIndType)!=CELL_DIM:
            # print("Error: cell dimension must be a non-negative integer!")
        # assert(np.rint(CELL_DIM).astype(SmallIndType)==CELL_DIM)

        if name is not None:
            name_str = isinstance(name, str)
            if not name_str:
                print("Error: name is not a string!")
            assert(name_str)

        if mesh is not None:
            mesh_correct = isinstance(mesh, SimplexMesh)
            if not mesh_correct:
                print("Error: mesh is not of type SimplexMesh!")
            assert(mesh_correct)

        self.name       = name
        self.mesh       = mesh
        self.degree     = None
        #self.basis_func = None
        self.num_basis  = None
        # Nodal_Data        % nodal variable data for all DoFs
        # Nodal_Top         % the nodal DoF topological arrangement
        self.DoFmap     = None
        self.free_DoFs  = None
        self.fixed_DoFs = None

    def __str__(self):
        if self.degree is not None:
            FE_deg = self.degree
        else:
            FE_deg = -1
        FE_deg_STR = self.name + " Finite Element space of degree = " + str(self.degree)
        if self.DoFmap is not None:
            Num_DoF = self.DoFmap.size
        else:
            Num_DoF = 0
        Num_DoF_STR = "Total number of DoFs = " + str(Num_DoF)
        if self.fixed_DoFs is not None:
            Num_Fixed_DoF = self.fixed_DoFs.size
        else:
            Num_Fixed_DoF = 0
        Num_Fixed_DoF_STR = "Number of fixed DoFs = " + str(Num_Fixed_DoF)
        OUT_STR = FE_deg_STR + "\n" \
                  + Num_DoF_STR + "\n" \
                  + Num_Fixed_DoF_STR + "\n"
        return OUT_STR

    # def Clear(self):
        # """This clears all cell data.
         # The _size attribute (i.e. number of cells) is set to zero."""
        # del(self.vtx)
        # del(self.halffacet)
        # self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        # self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        # self._size = 0

    def Set_Mesh(self, mesh):
        """This sets the mesh that the finite element space is defined on."""
        self.mesh = mesh

        mesh_correct = isinstance(mesh, SimplexMesh)
        if not mesh_correct:
            print("Error: mesh is not of type SimplexMesh!")
        assert(mesh_correct)
        
        self.mesh = mesh
        # the DoFs are now invalid, so clear it
        self.DoFmap     = None
        self.free_DoFs  = None
        self.fixed_DoFs = None

    def Set_fixed_DoFs(self, fixed_DoFs):
        """This sets the fixed DoFs of the finite element space."""

        # make sure they are in the valid range
        max_DoF = np.max(self.DoFmap)
        min_DoF = 0
        
        fixed_DoFs.shape = (fixed_DoFs.size, )
        
        min_valid_DoFs = np.min(fixed_DoFs) >= min_DoF
        max_valid_DoFs = np.max(fixed_DoFs) <= max_DoF
        if not min_valid_DoFs:
            print("fixed_DoFs contain indices < 0!")
        assert(min_valid_DoFs)
        if not max_valid_DoFs:
            print("fixed_DoFs contain indices > max(DoFmap)!")
        assert(max_valid_DoFs)

        self.fixed_DoFs = fixed_DoFs
        All_DoFs = self.DoFmap.flatten()
        self.free_DoFs = np.setdiff1d(All_DoFs, self.fixed_DoFs)

    def Size(self):
        """This returns the total number of DoFs."""
        distinct_DoFs = np.unique(self.DoFmap)
        return distinct_DoFs.size

    # def Dim(self):
        # """This returns the topological dimension of the cells."""
        # return self._cell_dim




    #



class Lagrange(FEspace):
    """    
    A class for defining Lagrange finite element spaces.
    """

    def __init__(self, mesh, degree=1):
        super().__init__("Lagrange", mesh)
        
        # need to enforce that the degree >= 0
        degree_correct = degree >= 0
        if not degree_correct:
            print("Error: given degree must be >= 0 for Lagrange!")
        assert(degree_correct)

        if degree > 1:
            print("Error: not implemented!")

        self.degree = degree
        if self.degree==0:
            self.num_basis = 1
        elif self.degree==1:
            self.num_basis = self.mesh.Top_Dim() + 1
        else:
            print("Error: not implemented!")
            ERROR

    def __str__(self):
        OUT_STR = super().__str__()

        return OUT_STR

    def Create_DoFmap(self):
        """This creates the Degree-of-Freedom map for the finite element space
           on the given mesh.
        """
        
        if self.degree==0:
            NC = self.mesh.Num_Cell()
            self.DoFmap = np.arange(NC)
            self.DoFmap.shape = (NC, 1)
        elif self.degree==1:
            # this is assuming they are indexed from 0 to max_DoF,
            #      with no gaps.
            NC = self.mesh.Num_Cell()
            self.DoFmap = np.copy(self.mesh.Cell.vtx[0:NC,:])
        else:
            print("Error: not implemented!")
            ERROR

    def Eval_Basis_Func(self, points, deriv_str="val"):
        """Evaluate the basis functions (BFs) at the given points on the unit (reference) simplex.

        Inputs: points: (M,TD) numpy array of points to evaluate basis functions at;
                         the points should be inside the unit simplex.
                deriv_str: string equal to "val" or "grad".

        If deriv_str=="val":
        Output: Basis_Eval: (B,M) numpy array of basis function values at the M points;
                             B = number of basis functions.
        If deriv_str=="grad":
        Output: Basis_Eval: (B,M,TD) numpy array of basis function gradient values at the M points;
                             Ex: at kk point, Basis_Eval[ii,kk,:] is the gradient of the ii BF.
        """
        if type(points) is not np.ndarray:
            print("Error: points must be a numpy array!")
            return

        Num_Pts = points.shape[0]
        Pt_dim = points.shape[1]
        if Pt_dim != self.mesh.Top_Dim():
            print("Error: the dimension of the points must match the mesh!")
            return

        if deriv_str == "val":
            # init
            Basis_Eval = np.zeros((self.num_basis,Num_Pts), dtype=RealType)
            if self.degree==0:
                Basis_Eval[0,:] = 1.0
            if self.degree==1:
                # \phi_0
                Pt_sum = np.sum(points, axis=1)
                Basis_Eval[0,:] = 1.0 - Pt_sum[:]
                # \phi_1, ..., \phi_TD
                pts_transpose = np.transpose(points)
                Basis_Eval[1:,:] = pts_transpose[:,:]
            else:
                print("Error: not implemented!")
        elif deriv_str == "grad":
            # init
            Basis_Eval = np.zeros((self.num_basis,Num_Pts,Pt_dim), dtype=RealType)
            # note: gradient of constant basis function is zero already
            if self.degree==1:
                # gradient of degree 1 polys is constant
                # \phi_0
                Basis_Eval[0,:] = -1.0
                # \phi_1, ..., \phi_TD
                for ii in range(1,self.num_basis):
                    Basis_Eval[ii,:,ii-1] = 1.0
            else:
                print("Error: not implemented!")
        else:
            print("Error: not implemented!")

        return Basis_Eval

class FEmatrix:
    """
    Class for computing finite element matrices.

    Note: ???
    """

    def __init__(self, test_space, trial_space=None):
        """
        ?????? CELL_DIM >= 0 is the topological dimension that the cells live in.
        res_buf in [0.0, 1.0] is optional, and is for reserving extra space for cell data.
        """
        # if (CELL_DIM<0):
            # print("Error: cell dimension must be non-negative!")
        # assert(CELL_DIM>=0)
        # if np.rint(CELL_DIM).astype(SmallIndType)!=CELL_DIM:
            # print("Error: cell dimension must be a non-negative integer!")
        # assert(np.rint(CELL_DIM).astype(SmallIndType)==CELL_DIM)

        is_test = isinstance(test_space, FEspace)
        if not is_test:
            print("Error: test_space is not a FEspace!")
        assert(is_test)

        if trial_space is not None:
            is_trial = isinstance(trial_space, FEspace)
            if not is_trial:
                print("Error: trial_space is not a FEspace!")
            assert(is_trial)

        self.test_space = test_space
        if trial_space is not None:
            self.trial_space = trial_space
        else:
            self.trial_space = test_space

        # make sure the two spaces are defined on the same mesh
        # simple check for now
        same_num_cells = self.test_space.DoFmap.shape[0] == self.trial_space.DoFmap.shape[0]
        if not same_num_cells:
            print("Error: test_space and trial_space must have same number of cells!")
        assert(same_num_cells)

        # get matrix dims
        self.num_rows = self.test_space.Size()
        self.num_cols = self.trial_space.Size()

    # def __str__(self):
        # if self.degree is not None:
            # FE_deg = self.degree
        # else:
            # FE_deg = -1
        # FE_deg_STR = self.name + " Finite Element space of degree = " + str(self.degree)
        # if self.DoFmap is not None:
            # Num_DoF = self.DoFmap.size
        # else:
            # Num_DoF = 0
        # Num_DoF_STR = "Total number of DoFs = " + str(Num_DoF)
        # if self.fixed_DoFs is not None:
            # Num_Fixed_DoF = self.fixed_DoFs.size
        # else:
            # Num_Fixed_DoF = 0
        # Num_Fixed_DoF_STR = "Number of fixed DoFs = " + str(Num_Fixed_DoF)
        # OUT_STR = FE_deg_STR + "\n" \
                  # Num_DoF_STR + "\n" \
                  # Num_Fixed_DoF_STR + "\n" \
        # return OUT_STR

    # def Clear(self):
        # """This clears all cell data.
         # The _size attribute (i.e. number of cells) is set to zero."""
        # del(self.vtx)
        # del(self.halffacet)
        # self.vtx = np.full((1, self._cell_dim+1), NULL_Vtx)
        # self.halffacet = np.full((1, self._cell_dim+1), NULL_HalfFacet)
        # self._size = 0

    def Mass_Matrix(self):
        """This outputs a sparse matrix representing the standard mass matrix."""

        quad_deg = self.test_space.degree + self.trial_space.degree
        quad_deg = np.max(np.array([quad_deg, 1]))
        TD = self.test_space.mesh.Top_Dim()
        X, W = sm.Unit_Simplex_Quadrature(quad_deg, TD)
        # X is in barycentric coordinates, so remove the first
        X = np.delete(X, 0, 1)

        test_phi  = self.test_space.Eval_Basis_Func(X, deriv_str="val")
        trial_phi = self.trial_space.Eval_Basis_Func(X, deriv_str="val")
        
        # compute local FE matrix
        
        # \hat{phi}[ii,:] * W[:], for all basis func indices ii
        # print(X)
        # print(test_phi)
        # print(W)
        test_phi_with_weights = np.multiply(test_phi, W)
        trial_phi_trans = np.transpose(trial_phi)
        # dot(\hat{phi}_W[ii,:],\hat{phi}[jj,:]) =
        #     \int_{\hat{T}} \hat{phi}_{ii} \hat{phi}_{jj} d\hat{x}, for all ii, jj
        M_loc = np.matmul(test_phi_with_weights,trial_phi_trans)
        M_loc_flat = M_loc.flatten()
        
        # compute all affine maps of the mesh (get Jacobians)
        # compute the det(Jacobian)
        Fact0 = math.factorial(TD)
        det_Jac = Fact0 * self.test_space.mesh.Volume()
        det_Jac.shape = (det_Jac.size, 1)
        
        # create the COO format
        test_DoFmap  = np.kron(self.test_space.DoFmap, np.ones((1,self.trial_space.DoFmap.shape[1])))
        trial_DoFmap = np.kron(np.ones((1,self.test_space.DoFmap.shape[1])), self.trial_space.DoFmap)
        # this now includes the det(Jac) factor so that we compute:
        #      \int_{T} phi_{ii} phi_{jj} dx,  for all ii, jj
        M_loc_rep    = np.kron(det_Jac, M_loc_flat)

        ii_vec = test_DoFmap.flatten()
        jj_vec = trial_DoFmap.flatten()
        ss_vec = M_loc_rep.flatten()
        CM = sp.sparse.coo_matrix((ss_vec, (ii_vec, jj_vec)), shape=(self.num_rows, self.num_cols))

        # convert to sparse format
        Sparse_Mass = CM.tocsr()
        return Sparse_Mass

    def Lumped_Mass_Matrix(self):
        """This outputs a sparse (diagonal) matrix that is a lumped mass matrix."""

        TD = self.test_space.mesh.Top_Dim()
        
        # n-dim volumes of each cell
        Vol = self.test_space.mesh.Volume()
        
        Num_Vtx = self.test_space.mesh._Vtx.Size()
        Vtx_Star_Vol = np.zeros(Num_Vtx)
        for kk in range(Num_Vtx):
            vertexAttachments = self.test_space.mesh.Get_Cells_Attached_To_Vertex(kk)
            star_vols = Vol[vertexAttachments]
            Vtx_Star_Vol[kk] = np.sum(star_vols)

        Sparse_LM = sp.sparse.diags((1/(TD+1))*Vtx_Star_Vol, 0, shape=(Num_Vtx,Num_Vtx))
        return Sparse_LM

    def Stiffness_Matrix(self):
        """This outputs a sparse matrix representing the standard stiffness matrix."""

        quad_deg = (self.test_space.degree-1) + (self.trial_space.degree-1)
        quad_deg = np.max(np.array([quad_deg, 1]))
        TD = self.test_space.mesh.Top_Dim()
        X, W = sm.Unit_Simplex_Quadrature(quad_deg, TD)
        # X is in barycentric coordinates, so remove the first
        X = np.delete(X, 0, 1)
        Num_Q_pts = W.size

        test_Gphi  = self.test_space.Eval_Basis_Func(X, deriv_str="grad")
        Num_Test = test_Gphi.shape[0]
        trial_Gphi = self.trial_space.Eval_Basis_Func(X, deriv_str="grad")
        Num_Trial = test_Gphi.shape[0]
        
        # get the Jacobians
        A, b = self.test_space.mesh.Affine_Map()
        Num_Cell = self.test_space.mesh.Num_Cell()
        # compute the metric
        A_tr = np.transpose(A, (0, 2, 1))
        g = np.matmul(A_tr,A)
        # inverse metric
        inv_g = np.linalg.inv(g)
        # inv_g is (M,TD,TD)
        det_g = np.linalg.det(g)
        # dS is (M,)
        dS = np.sqrt(det_g)

        # do small calculation
        local_test_mult_local_trial = np.zeros((Num_Test,Num_Trial,TD,TD))
        for ii in range(Num_Test):
            for jj in range(Num_Trial):
                for kk in range(Num_Q_pts):
                    # integrate
                    local_test_mult_local_trial[ii,jj,:,:] += W[kk] * np.outer(test_Gphi[ii,kk,:], trial_Gphi[jj,kk,:])

        # compute the integral for all elements (cells)
        temp_int = np.tensordot(inv_g, local_test_mult_local_trial, axes=([1,2],[2,3]))
        temp_int.shape = [Num_Cell, Num_Test*Num_Trial]

        # multiply by dS
        M_loc_all = (temp_int.transpose() * dS).transpose()
        # (M, Num_Test*Num_Trial)

        # create the COO format
        test_DoFmap  = np.kron(self.test_space.DoFmap, np.ones((1,self.trial_space.DoFmap.shape[1])))
        trial_DoFmap = np.kron(np.ones((1,self.test_space.DoFmap.shape[1])), self.trial_space.DoFmap)

        ii_vec = test_DoFmap.flatten()
        jj_vec = trial_DoFmap.flatten()
        ss_vec = M_loc_all.flatten()
        CS = sp.sparse.coo_matrix((ss_vec, (ii_vec, jj_vec)), shape=(self.num_rows, self.num_cols))

        # convert to sparse format
        Sparse_Stiff = CS.tocsr()
        return Sparse_Stiff
