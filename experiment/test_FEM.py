# BEGIN: imports

#import os
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../Newton_Methods')
#sys.path.insert(1, '../VTK')
# print(sys.path)

# SWW: keep this order of imports!

import numpy as np

# END: imports

#from ahf import *
#import ahf as AHF

#from ahf.SimplexMesh import *
from ahf.MeshFactory import *
#from ahf.fem.Formulation import *
import ahf.fem.Formulation as FEM

MF = MeshFactory()
print(MF)
VC, Mesh = MF.Simplex_Mesh_Of_Rectangle(Pll=[0.0, 0.0], Pur=[2.0, 1.0], N0=4, N1=3)
VC.Open()
VC.Change_Dimension(5)
Num_Vtx = Mesh.Cell.Num_Vtx()
VC.coord[0:Num_Vtx,2] = (VC.coord[0:Num_Vtx,0] - 1.0)**2 + (VC.coord[0:Num_Vtx,1] - 0.5)**2
VC.Close()
#print(VC)
print(Mesh)
#VC.Print()
#Mesh.Cell.Print()
#Mesh.Print_Vtx2HalfFacets()

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

Tangent_Star_tr = np.transpose(Tangent_Star, (0, 2, 1))

test_vi = 1
print("Tangent_Star_tr[test_vi,:,:]:")
print(Tangent_Star_tr[test_vi,:,:])

# print("U[0,:,:]:")
# print(U[0,:,:])
# print("S[0,:]:")
# print(S[0,:])
print("Ortho[test_vi,:,:]")
print(Ortho[test_vi,:,:])

print("Tangent_Star_tr[test_vi,:,0:6] DPs with Ortho[test_vi,:,[0,1,2]]:")
for kk in range(6):
    DP_0 = np.dot(Tangent_Star_tr[test_vi,:,kk],Ortho[test_vi,:,0]).item(0)
    DP_1 = np.dot(Tangent_Star_tr[test_vi,:,kk],Ortho[test_vi,:,1]).item(0)
    DP_2 = np.dot(Tangent_Star_tr[test_vi,:,kk],Ortho[test_vi,:,2]).item(0)
    print([DP_0, DP_1, DP_2])

# print(np.sum(Vh[0,0,:]**2))
# print(np.sum(Vh[0,1,:]**2))
