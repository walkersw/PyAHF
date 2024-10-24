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
