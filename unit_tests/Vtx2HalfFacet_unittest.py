import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.Vtx2HalfFacet_Mapping import *

class TestVtx2HalfFacet(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.V2HF = Vtx2HalfFacetMap()

    @classmethod
    def tearDownClass(self):
        self.V2HF.Clear()

    #def setUp(self):
        #self.V2HF = <do nothing>

    def tearDown(self):
        self.V2HF.Clear()

    def test_Size(self):
        self.V2HF.Reserve(10)
        self.assertEqual(self.V2HF.Size(), 0, "Size should be 0.")

    def test_Reserve(self):
        num_reserve = 3
        self.V2HF.Reserve(num_reserve)
        Reserved_Size = np.rint(np.ceil((1.0 + self.V2HF._reserve_buffer) * num_reserve))
        self.assertEqual(self.V2HF.VtxMap.size, Reserved_Size, "Reserved size should be " + str(Reserved_Size) + ".")

    def test_Get_Half_Facets(self):
        self.V2HF.Reserve(5)
        print(" ")
        
        hf = np.array((12, 2), dtype=HalfFacetType)
        self.V2HF.Append(10, hf)
        
        hf[['ci', 'fi']] = (8, 1)
        self.V2HF.Append(10, hf)

        print(self.V2HF)

        self.assertEqual(self.V2HF.VtxMap[2]==NULL_VtxHalfFacet, True, "Should be Null.")

        hf_ind0, hf_total = self.V2HF.Get_Half_Facets(10)
        for kk in range(hf_total):
            print("This is the attached vertex/half-facet: " + str(self.V2HF.VtxMap[hf_ind0 + kk]))

        self.assertEqual(hf_total, 2, "Should be 2.")
        
        hf0 = np.array((12, 2), dtype=HalfFacetType)
        hf1 = np.array((8, 1), dtype=HalfFacetType)
        self.assertEqual(self.V2HF.VtxMap[0][['ci', 'fi']]==hf0, True, "Should be (12, 2).")
        self.assertEqual(self.V2HF.VtxMap[1][['ci', 'fi']]==hf1, True, "Should be (8, 1).")

        VM = self.V2HF.Get_Half_Facets(10,"array")

        for vhf in VM:
            print("This is the attached vertex/half-facet: " + str(vhf))
        
        hf = self.V2HF.Get_Half_Facet(10)
        self.assertEqual(hf==hf0, True, "Should be (12, 2).")

        # now Sort(); this sorts based on vertex indices, followed by cell index, then facet index
        self.V2HF.Sort()
        hf = self.V2HF.Get_Half_Facet(10)
        self.assertEqual(hf==hf1, True, "Should be (8, 1).")

    def test_Display(self):
        self.V2HF.Reserve(5)
        print(" ")
        
        hf = np.array((4, 2), dtype=HalfFacetType)
        self.V2HF.Append(3, hf)
        hf[['ci', 'fi']] = (7, 0)
        self.V2HF.Append(1, hf)
        hf[['ci', 'fi']] = (9, 1)
        self.V2HF.Append(3, hf)
        hf[['ci', 'fi']] = (14, 2)
        self.V2HF.Append(1, hf)
        hf[['ci', 'fi']] = (11, 0)
        self.V2HF.Append(5, hf)

        self.V2HF.Display_Half_Facets()
        self.V2HF.Sort()
        self.V2HF.Display_Half_Facets()
        
        unique_vertices = self.V2HF.Get_Unique_Vertices()
        self.assertEqual( np.array_equal(unique_vertices,[1, 3, 5]), True, "Should be [1, 3, 5].")
        self.V2HF.Display_Unique_Vertices()




if __name__ == '__main__':
    unittest.main()
