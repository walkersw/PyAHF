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

from ahf.Vtx2HalfFacet_Mapping import *

# init output code
OUTPUT_CODE = 0 # 0 indicates success, > 0 is failure

# create an object
V2HF = Vtx2HalfFacetMap()

# reserve enough room for 5 vtx2half-facets
V2HF.Reserve(5)

if (V2HF.Size()!=0):
    print("Size of V2HF should still be zero.")
    OUTPUT_CODE = 1

print(V2HF.Size())

hf = np.array((12, 2), dtype=HalfFacetType)

V2HF.Append(10, hf)
hf[['ci', 'fi']] = (8, 1)

print(hf)

V2HF.Append(10, hf)

print(V2HF)

hf_ind0, hf_total = V2HF.Get_Half_Facets(10)

for kk in range(hf_total):
    print("This is the attached vertex/half-facet: " + str(V2HF.VtxMap[hf_ind0 + kk]))

VM = V2HF.Get_Half_Facets(10,"array")

for vhf in VM:
    print("This is the attached vertex/half-facet: " + str(vhf))

hf = V2HF.Get_Half_Facet(10)

print(hf)

V2HF.Display_Half_Facets()


TF = V2HF.VtxMap[1]==NULL_VtxHalfFacet
print(TF)

V2HF.Display_Unique_Vertices()
