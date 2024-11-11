# BEGIN: imports

#import os
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../Newton_Methods')
#sys.path.insert(1, '../VTK')
# print(sys.path)

# SWW: keep this order of imports!

import matplotlib.pyplot as plt
#import matplotlib.tri as tri
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

# END: imports

#from ahf import *
#import ahf as AHF

#from ahf.SimplexMesh import *
from ahf.MeshFactory import *
#from ahf.fem.Formulation import *
import ahf.fem.Formulation as FEM

MF = MeshFactory()
print(MF)
VC, Mesh = MF.Simplex_Mesh_Of_Rectangle(Pll=[0.0, 0.0], Pur=[2.0, 1.0], N0=30, N1=15, UseBCC=True)
VC.Open()
VC.Change_Dimension(3)
GD = VC.Dim()
Num_Vtx = Mesh.Cell.Num_Vtx()
Num_Cell = Mesh.Num_Cell()
#VC.coord[0:Num_Vtx,2] = (VC.coord[0:Num_Vtx,0] - 1.0)**2 + (VC.coord[0:Num_Vtx,1] - 0.5)**2
VC.coord[0:Num_Vtx,2] = np.sin(3*VC.coord[0:Num_Vtx,0] + 0.7) * np.cos(4*VC.coord[0:Num_Vtx,1])
VC.Close()
#print(VC)
print(Mesh)
#VC.Print()
#Mesh.Cell.Print()
#Mesh.Print_Vtx2HalfFacets()

Fixed_DoFs = Mesh.Cell.Get_FreeBoundary(get_vertices=True)

# Create a figure and an axes object.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface.
ax.plot_trisurf(VC.coord[0:Num_Vtx,0], VC.coord[0:Num_Vtx,1], VC.coord[0:Num_Vtx,2], triangles=Mesh.Cell.vtx[0:Num_Cell,:], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')
plt.title('Triangulated Surface Mesh')
plt.show()
#input("Press Enter to continue...")

FES = FEM.Lagrange(Mesh,degree=1)

FES.Create_DoFmap()

#print(FES.Size())
#print(FES.DoFmap)

FE_mat = FEM.FEmatrix(FES,FES)

#M = FE_mat.Mass_Matrix()
M = FE_mat.Lumped_Mass_Matrix()
K = FE_mat.Stiffness_Matrix()

nr = M.shape[0]
nc = M.shape[1]

#print(M.shape)
#print(M)

one_vec = np.ones((nr,))

M_one = M*one_vec
#print(M_one)
#print(one_vec)

val = np.matmul(one_vec, M_one).item(0)
print("Integral of 1.0 on mesh:")
print(val)
#print("{:2.10f}".format(val))


K_one = K*one_vec
#print(K_one)
#print(one_vec)

val_2 = np.matmul(one_vec, K_one).item(0)
print("Integral of |grad(1.0)|^2 on mesh:")
print(val_2)
#print("{:2.10f}".format(val_2))




# check vertex based tangents

# # get all mesh edges
# EE = Mesh.Cell.Get_Edges()

# # figure out the maximum number of edges per vertex
# Max_Edge_in_Star = 0
# Vtx_Star_temp = []
# for ii in np.arange(Num_Vtx, dtype=VtxIndType):
    # EE[:]['v0']
    # edge_indices_0 = np.argwhere(EE[:]['v0']==ii)
    # edge_indices_1 = np.argwhere(EE[:]['v1']==ii)
    # edge_ind = np.vstack((edge_indices_0,edge_indices_1))
    # #print(edge_ind[:,0])
    # Max_Edge_in_Star = np.max([Max_Edge_in_Star, edge_ind.size])
    # Vtx_Star_temp.append(edge_ind[:,0])

# #print(Vtx_Star_temp)

# # get data structure that maps vertices to attached edges (indices)
# # make there be the same number of edges per vertex, by having dummy edges
# # that will be zero vectors.


# # make a pure numpy version...
# np_EE = np.zeros((EE.size+1,2), dtype=VtxIndType)
# np_EE[:-1,0] = EE[:]['v0']
# np_EE[:-1,1] = EE[:]['v1']

# # add a fake edge
# np_EE[-1,0] = 0
# np_EE[-1,1] = 0
# fake_edge_index = np_EE.shape[0]-1

# print(np_EE)

# # now remake a more efficient data structure
# Vtx_Edge_Star = np.full((Num_Vtx,Max_Edge_in_Star), fake_edge_index, dtype=VtxIndType)
# for ii in np.arange(Num_Vtx, dtype=VtxIndType):
    # Edges_Attached_to_V_ii = Vtx_Star_temp[ii]
    # Num_Edges = Edges_Attached_to_V_ii.size
    # Vtx_Edge_Star[ii,0:Num_Edges] = Edges_Attached_to_V_ii[:]


Mesh_Edges, Vtx2Edge = Mesh.Get_Vtx_Edge_Star(None, efficient=True)

print("Mesh_Edges:")
print(Mesh_Edges)

print("Vtx2Edge:")
print(Vtx2Edge)


#TD = Mesh.Top_Dim()


# uv = Mesh.Cell.Get_Unique_Vertices()
# print("uv:")
# print(uv)

Vtx2Cell = Mesh.Get_Vtx_Cell_Attachments(None, efficient=True)

print("Vtx2Cell:")
print(Vtx2Cell)

# init the star tangent vector variable


# # compute the current edge vectors (Num_Edges,GD)
# Edge_Vec = Mesh._Vtx.coord[np_EE[:,1]][:] - Mesh._Vtx.coord[np_EE[:,0]][:]

# print("Edge_Vec:")
# print(Edge_Vec)


# GD = Mesh._Vtx.Dim()
# Tangent_Star = np.zeros((Num_Vtx,Max_Edge_in_Star,GD), dtype=CoordType)
# Tangent_Star[:,:,:] = Edge_Vec[Vtx_Edge_Star[:,:],:]

# print("Tangent_Star:")
# print(Tangent_Star)

Ortho, Edge_Vec, Tangent_Star = Mesh.Get_Vtx_Based_Orthogonal_Frame((Mesh_Edges, Vtx2Edge), frame_type="all", debug=True)

print("Edge_Vec:")
print(Edge_Vec)

print("Tangent_Star:")
print(Tangent_Star)

Ortho_ALT, CB_Star = Mesh.Get_Vtx_Based_Orthogonal_Frame_ALT(Vtx2Cell, frame_type="all", svd_with="tangent", debug=True)

print("CB_Star:")
print(CB_Star)

print("Ortho:")
print(Ortho)

print("Ortho_ALT:")
print(Ortho_ALT)

normal_vecs = Mesh.Get_Vtx_Averaged_Normal_Vector(Vtx2Cell)

print("normal_vecs:")
print(normal_vecs)

DP_NV     = np.abs( np.sum(Ortho[:,2,:] * normal_vecs[:,:], axis=1) )
DP_NV_ALT = np.abs( np.sum(Ortho_ALT[:,2,:] * normal_vecs[:,:], axis=1) )
#DP_NV_ALT = np.abs(np.dot(Ortho_ALT[:,2,:],normal_vecs[:,:]))

print("DP_NV:")
print(DP_NV)

print("DP_NV_ALT:")
print(DP_NV_ALT)


VI_chk = 17
print("Check vtx #" + str(VI_chk) + ":")

print("Tangent_Star[" + str(VI_chk) + ",:,:]:")
print(Tangent_Star[VI_chk,:,:])

print("manual way gives:")
print(Edge_Vec[Vtx2Edge[VI_chk,:],:])

DIFF = np.max(np.max(np.abs(Tangent_Star[VI_chk,:,:] - Edge_Vec[Vtx2Edge[VI_chk,:],:])))
print("Difference is:")
print(DIFF)


# # now do a stack of SVDs
# U, S, Vh = np.linalg.svd(Tangent_Star, full_matrices=False)

# print("The SVD of Tangent_Star:")
# print("U:")
# print(U)
# print("S:")
# print(S)
# print("Vh")
# print(Vh)
# you want the first TD rows of Vh...  that is the tangent space

# the first TD cols of Ortho is the tangent space

Tangent_Star_tp = np.transpose(Tangent_Star, (0, 2, 1))

test_vi = 3
print("Tangent_Star_tp[test_vi,:,:]:")
print(Tangent_Star_tp[test_vi,:,:])

# print("U[0,:,:]:")
# print(U[0,:,:])
# print("S[0,:]:")
# print(S[0,:])
print("Ortho[test_vi,:,:]")
print(Ortho[test_vi,:,:])

print("Tangent_Star_tp[test_vi,:,0:6] DPs with Ortho[test_vi,[0,1,2],:]:")
for kk in range(6):
    DP_0 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho[test_vi,0,:]).item(0)
    DP_1 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho[test_vi,1,:]).item(0)
    DP_2 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho[test_vi,2,:]).item(0)
    print([DP_0, DP_1, DP_2])

print("Tangent_Star_tp[test_vi,:,0:6] DPs with Ortho_ALT[test_vi,[0,1,2],:]:")
for kk in range(6):
    DP_0 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho_ALT[test_vi,0,:]).item(0)
    DP_1 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho_ALT[test_vi,1,:]).item(0)
    DP_2 = np.dot(Tangent_Star_tp[test_vi,:,kk],Ortho_ALT[test_vi,2,:]).item(0)
    print([DP_0, DP_1, DP_2])

# print(np.sum(Vh[0,0,:]**2))
# print(np.sum(Vh[0,1,:]**2))

# Ortho[test_vi,:,0] *
# T0 = Ortho[:,:,0]
# T0.shape = [Num_Vtx, GD, 1]

print(Ortho.shape)

NumBasis = 2
TanBasis = Ortho[:,0:NumBasis,:]
TanBasis.shape = [Num_Vtx, NumBasis, GD, 1]
TanBasis_tp = np.transpose(TanBasis, (0, 1, 3, 2))

Proj = np.matmul(TanBasis, TanBasis_tp)

print(Proj[0,0,:,:])

print("test this:")
P00 = np.outer(Ortho[0,0,:],Ortho[0,0,:])
print(P00)

# sum over the basis projections
Tangent_Proj = np.sum(Proj[:,0:NumBasis,:,:], axis=1)

# shape is [Num_Vtx, GD, GD]

print("Tangent_Proj:")
print(Tangent_Proj[3,:,:])

# need the normal projection
eye_stack = np.tile(np.eye(GD), (Num_Vtx, 1, 1))

Normal_Proj = eye_stack - Tangent_Proj

# TT[vi,:,:]

i_temp = np.arange(GD*Num_Vtx, dtype=VtxIndType)
rep_vec = np.ones(GD, dtype=VtxIndType)
i_vec = np.kron(rep_vec,i_temp)

#print("i_vec:")
#print(i_vec)

j_temp = np.arange(Num_Vtx, dtype=VtxIndType)
j1 = np.kron(rep_vec,j_temp)

shift_vec = np.arange(GD, dtype=VtxIndType)
shift_vec.shape = [GD,1]

j_mat = j1+(Num_Vtx*shift_vec)
j_vec = j_mat.flatten()

#print("j_vec:")
#print(j_vec)

s_val = Normal_Proj.flatten('F')

# this is the projection onto the normal space
Proj_Sparse = sp.sparse.csr_matrix((s_val, (i_vec, j_vec)), shape=(GD*Num_Vtx, GD*Num_Vtx))

print("Proj_Sparse")
print(Proj_Sparse)

# now assemble the diagonal lumped mass matrix, and repeat it GD (block) times
# assemble the stiffness matrix, and repeat it GD (block) times

sM = FE_mat.Lumped_Mass_Matrix()
sK = FE_mat.Stiffness_Matrix()

# print("M:")
# print(sM)

sK_tilde = sK.copy()
sK_diag = sK.diagonal()

sK_diag[Fixed_DoFs] = 1E100
sK_tilde.setdiag(sK_diag, k=0)

II = sp.sparse.identity(GD, format='csr')
vM = sp.sparse.kron(II, sM, format='csr')
vK_tilde = sp.sparse.kron(II, sK_tilde, format='csr')

M_1 = vM * Proj_Sparse
M_2 = Proj_Sparse * vM

# check the difference
ERR_MAT = M_1 - M_2
EM = np.max(np.abs(ERR_MAT))

print("ERROR:")
print(EM)

R1_array = -sK*VC.coord[0:Num_Vtx,:]
R1_array[Fixed_DoFs,:] = 0.0
#print(R1_array)
#R1_vec = R1_array.flatten('F')
R1_vec = np.reshape(R1_array, (Num_Vtx*GD,), order='F', copy=True)

#print(R1_vec)
R2_vec = np.zeros(R1_vec.size)
RHS = np.concatenate((R1_vec,R2_vec))

# form the linear system
# [\tilde{vK},       vM*vP] [\delta X] = [-vK*X_old]
# [vP*vM, -\tau * vP*vM*vP] [ \kappa ] = [0]

tau = 0.001
C = Proj_Sparse * vM * Proj_Sparse
MAT = sp.sparse.bmat([[vK_tilde, vM * Proj_Sparse], [Proj_Sparse * vM, -tau * C]], format='csr')


# flatten the coordinate vector

#numpy.reshape(a, /, shape=None, *, newshape=None, order='C', copy=None)


# solve for update
#scipy.sparse.linalg.LinearOperator
#scipy.sparse.linalg.cg
#x = sp.sparse.linalg.spsolve(A, b)

SOLN = sp.sparse.linalg.spsolve(MAT, RHS)

delta_X_vec = SOLN[0:Num_Vtx*GD]
kappa_vec   = SOLN[Num_Vtx*GD:]

# unflatten...

delta_X = np.reshape(delta_X_vec, (Num_Vtx,GD), order='F', copy=True)
kappa   = np.reshape(kappa_vec, (Num_Vtx,GD), order='F', copy=True)

print("delta_X:")
print(delta_X)
print("kappa:")
print(kappa)

# alternative way
# [\tilde{vK} + (1/\tau) * vP*vM*vP] [\delta X] = [-vK*X_old]
MAT_alt = vK_tilde + (1/tau) * C

delta_X_vec_alt = sp.sparse.linalg.spsolve(MAT_alt, R1_vec)
delta_X_alt = np.reshape(delta_X_vec_alt, (Num_Vtx,GD), order='F', copy=True)

delta_X_DIFF = np.abs(delta_X_alt - delta_X)
delta_X_ERR = np.max(delta_X_DIFF)
print("delta_X_ERR:")
print(delta_X_ERR)
print(delta_X_DIFF)

# update coordinates...


# define time interval and number of time-steps
Final_t = 4*0.5
Num_Steps = 4*20
# compute time-step
tau = Final_t / Num_Steps

Num_Steps = 80*25
Num_Steps = 950
tau = 0.001
# 0.025

# start the time-stepping loop here
for ti in range(Num_Steps):
    print("\rti = {:2G}".format(ti), end='')

    #Ortho, Edge_Vec, Tangent_Star = Mesh.Get_Vtx_Based_Orthogonal_Frame((Mesh_Edges, Vtx2Edge), frame_type="all", debug=True)

    Ortho, CB_Star = Mesh.Get_Vtx_Based_Orthogonal_Frame_ALT(Vtx2Cell, frame_type="all", svd_with="tangent", debug=True)

    normal_vecs_array = Mesh.Get_Vtx_Averaged_Normal_Vector(Vtx2Cell)
    # (NV,GD)
    normal_vec = np.reshape(normal_vecs_array, (Num_Vtx, GD, 1))
    normal_vec_tp = np.transpose(normal_vec, (0, 2, 1))
    
    # Normal_Proj = np.matmul(normal_vec, normal_vec_tp)
    # # shape is [Num_Vtx, GD, GD]

    NumBasis = 2
    TanBasis = Ortho[:,0:NumBasis,:]
    TanBasis.shape = [Num_Vtx, NumBasis, GD, 1]
    TanBasis_tp = np.transpose(TanBasis, (0, 1, 3, 2))

    Proj = np.matmul(TanBasis, TanBasis_tp)

    # sum over the basis projections
    Tangent_Proj = np.sum(Proj[:,0:NumBasis,:,:], axis=1)
    # shape is [Num_Vtx, GD, GD]

    # need the normal projection
    eye_stack = np.tile(np.eye(GD), (Num_Vtx, 1, 1))

    Normal_Proj = eye_stack - Tangent_Proj

    # create the sparse projection matrix

    i_temp = np.arange(GD*Num_Vtx, dtype=VtxIndType)
    rep_vec = np.ones(GD, dtype=VtxIndType)
    i_vec = np.kron(rep_vec,i_temp)

    j_temp = np.arange(Num_Vtx, dtype=VtxIndType)
    j1 = np.kron(rep_vec,j_temp)

    shift_vec = np.arange(GD, dtype=VtxIndType)
    shift_vec.shape = [GD,1]

    j_mat = j1+(Num_Vtx*shift_vec)
    j_vec = j_mat.flatten()

    s_val = Normal_Proj.flatten('F')

    # this is the projection onto the normal space
    Proj_Sparse = sp.sparse.csr_matrix((s_val, (i_vec, j_vec)), shape=(GD*Num_Vtx, GD*Num_Vtx))
    
    #print("Proj_Sparse:")
    #print(Proj_Sparse)
    
    # now assemble the diagonal lumped mass matrix, and repeat it GD (block) times
    # assemble the stiffness matrix, and repeat it GD (block) times

    sM = FE_mat.Lumped_Mass_Matrix()
    sK = FE_mat.Stiffness_Matrix()

    sK_tilde = sK.copy()
    sK_diag = sK.diagonal()

    sK_diag[Fixed_DoFs] = 1E100
    sK_tilde.setdiag(sK_diag, k=0)

    II = sp.sparse.identity(GD, format='csr')
    vM = sp.sparse.kron(II, sM, format='csr')
    vK_tilde = sp.sparse.kron(II, sK_tilde, format='csr')
    vK = sp.sparse.kron(II, sK, format='csr')

    R1_array = -sK*VC.coord[0:Num_Vtx,:]
    R1_array[Fixed_DoFs,:] = 0.0
    #print(R1_array)
    #R1_vec = R1_array.flatten('F')
    R1_vec = np.reshape(R1_array, (Num_Vtx*GD,), order='F', copy=True)

    # alternative way
    # [\tilde{vK} + (1/\tau) * vP*vM*vP] [\delta X] = [-vK*X_old]
    C = Proj_Sparse * vM * Proj_Sparse
    MAT_alt = vK_tilde + (1/tau) * C
    
    # print("MAT:")
    # print(MAT_alt)

    delta_X_vec = sp.sparse.linalg.spsolve(MAT_alt, R1_vec)
    delta_X = np.reshape(delta_X_vec, (Num_Vtx,GD), order='F', copy=True)

    # compute the curvature vector??
    # \kappa_vec = -(1/\tau)*vP*delta_X_vec
    kappa_vec = -(1/tau) * Proj_Sparse * delta_X_vec
    kappa = np.reshape(kappa_vec, (Num_Vtx,GD), order='F', copy=True)

    # update coordinates
    VC.Open()
    VC.coord[0:Num_Vtx,:] += delta_X
    VC.Close()

#
print(' ')

# Create a figure and an axes object.
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# plot the surface
ax2.plot_trisurf(VC.coord[0:Num_Vtx,0], VC.coord[0:Num_Vtx,1], VC.coord[0:Num_Vtx,2], triangles=Mesh.Cell.vtx[0:Num_Cell,:], cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_aspect('equal')
plt.title('surface at later time')
plt.show()

input("Press Enter to continue...")