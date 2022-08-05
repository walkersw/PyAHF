# BEGIN: imports

import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../Newton_Methods')
#sys.path.insert(1, '../VTK')
# print(sys.path)

# SWW: keep this order of imports!

import numpy as np

# END: imports

#from ahf import *
import ahf as AHF

V2HF = AHF.Vtx2HalfFacetMap()

V2HF.Reserve(10)

print(VSHF.Size())



# // unit test
# int main()
# {
    # // init output code
    # int OUTPUT_CODE = 0; // 0 indicates success, > 0 is failure

    # // minor test
    # Vtx2HalfFacet_Mapping  V2HF;
    # HalfFacetType  hf;
    # hf.Set(12,2);
    # V2HF.Append(10, hf);
    # hf.Set(8,1);
    # V2HF.Append(10, hf);
    # V2HF.Get_Half_Facet(10, hf);
    # cout << "This is the retrieved half-facet: <" << hf.ci << ", " << hf.fi << ">." << endl;
    # cout << endl;

    # // check retrieved half-facet against reference data
    # HalfFacetType attached1_REF[1];
    # attached1_REF[0].Set(12,2);
    # // you should get the first one that was "Appended"
    # if (!hf.Equal(attached1_REF[0]))
    # {
        # cout << "Retrieved HalfFacet data is incorrect!" << endl;
        # OUTPUT_CODE = 1;
    # }

    # // now sort
    # V2HF.Sort();
    # V2HF.Get_Half_Facet(10, hf);
    # cout << "This is the retrieved half-facet after sorting: <" << hf.ci << ", " << hf.fi << ">." << endl;
    # cout << endl;
    # // check retrieved half-facet against reference data
    # // you should now get the other one that was "Appended"
    # //     b/c we sort on vertex indices, then on the half-facet.
    # HalfFacetType attached2_REF[1];
    # attached2_REF[0].Set(8,1);
    # if (!hf.Equal(attached2_REF[0]))
    # {
        # cout << "Retrieved HalfFacet data is incorrect!" << endl;
        # OUTPUT_CODE = 2;
    # }

    # if (OUTPUT_CODE==0)
        # cout << "Unit test is successful!" << endl;
    # else
        # cout << "Unit test failed!" << endl;
    # cout << endl;

    # return OUTPUT_CODE;
# }


