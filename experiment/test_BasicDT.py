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

from ahf.BasicClasses import *

# create an object
Cell = CellSimplexType(2)

# reserve enough room for 5 cells
Cell.Reserve(5)

print(Cell.Size())


Cell.Append_Cell([1, 2, 3])

Cell.Append_Cell([4, 23])


print(Cell.vtx)
print(Cell.halffacet)

print(Cell)

new_cell_vtx = [2, 5, 4, 6, 88, 9, 1, 4, 90, 23, 45, 74]

Cell.Append_Cell_Data(4, new_cell_vtx)

print(Cell.vtx)
print(Cell.halffacet)

print(Cell)


# hf_ind0, hf_total = V2HF.Get_Half_Facets(10)

# for kk in range(hf_total):
    # print("This is the attached vertex/half-facet: " + str(V2HF.VtxMap[hf_ind0 + kk]))

# VM = V2HF.Get_Half_Facets(10,"array")

# for vhf in VM:
    # print("This is the attached vertex/half-facet: " + str(vhf))

# hf = V2HF.Get_Half_Facet(10)

# print(hf)

# V2HF.Display_Half_Facets()


# TF = V2HF.VtxMap[1]==NULL_VtxHalfFacet
# print(TF)

# V2HF.Display_Unique_Vertices()
