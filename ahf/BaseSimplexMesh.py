"""
ahf.BaseSimplexMesh.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base class for array based half-facet (AHF) data structure to store and
process simplex meshes.

Also, see "Vtx2HalfFacet_Mapping.py" for more explanation.

Copyright (c) 08-12-2022,  Shawn W. Walker
"""

import numpy as np
# from ahf import SmallIndType, MedIndType, VtxIndType, CellIndType
# from ahf import NULL_Small, NULL_Med, NULL_Vtx, NULL_Cell
# from ahf import RealType, PointType
from ahf import *

# define data types

# half-facet
HalfFacetType = np.dtype({'names': ['ci', 'fi'],
                          'formats': [CellIndType, SmallIndType],
                          'titles': ['cell index', 'local facet (entity) index']})
NULL_HalfFacet = np.array((NULL_Cell, NULL_Small), dtype=HalfFacetType)

# vertex half-facet
VtxHalfFacetType = np.dtype({'names': ['vtx', 'ci', 'fi'],
                             'formats': [VtxIndType, CellIndType, SmallIndType],
                             'titles': ['global vertex index', 'cell index', 'local facet (entity) index']})
NULL_VtxHalfFacet = np.array((NULL_Vtx, NULL_Cell, NULL_Small), dtype=VtxHalfFacetType)

FIX!!!!


class BaseSimplexMesh:
    """
    Base class for array based half-facet (AHF) data structure to store and process meshes.
    This base class is used in deriving the 0-D, 1-D, 2-D, and 3-D mesh classes,
    as well as arbitrarily higher dimensions (all *simplex* meshes).
    
    Note: no vertex coordinates are stored in this class; this is purely topological.

   Scheme for ordering *local* topological entities
   ------------------------------------------------

   RULE: local "facet" Fi is always *opposite* local vertex Vi.

   DIM=1. Interval: facets are vertices (points)

        V0 +-------------------+ V1

   i.e. "facet" F0 = V1 (which is opposite V0), and "facet" F1 = V0 (which is opposite V1)

   DIM=2. Triangle: facets are edges (line segments)

        V2 +
           |\
           |  \
           |    \
           |      \
           |        \  E0
        E1 |          \
           |            \
           |              \
           |                \
           |                  \
        V0 +-------------------+ V1
                    E2

   i.e. "facet" F0 = E0 (which is opposite V0), "facet" F1 = E1 (which is opposite V1), etc...

   DIM=3: Tetrahedron: facets are faces (triangles)

                 V3 +
                   /|\
                  / | \
                 |  |  \
                |   |   \
               |    |    \
               |    |  F1 \               F0 (opposite V0)
              | F2  |      \
              |     |       \
             |   V0 +--------+ V2
             |     /      __/
            |    /  F3 __/
            |  /    __/
           | /   __/
          |/  __/
       V1 +--/

   i.e. facet F0 = [V1, V2, V3] (which is opposite V0),
        facet F1 = [V0, V3, V2] (which is opposite V1),
        facet F2 = [V0, V1, V3] (which is opposite V2),
        facet F3 = [V0, V2, V1] (which is opposite V3).

   Higher DIM: the pattern continues...

   EXAMPLE:  Mesh of two triangles.  In this case, a half-facet == half-edge.

        V3 +-------------------+ V2
           |\                  |
           |  \                |
           |    \        T1    |
           |      \            |
           |        \          |
           |          \        |
           |            \      |
           |     T0       \    |
           |                \  |
           |                  \|
        V0 +-------------------+ V1

   Triangle Connectivity and Sibling Half-Facet (Half-Edge) Data Struct:

   triangle |   vertices   |     sibling half-edges
    indices |  V0, V1, V2  |     E0,     E1,      E2
   ---------+--------------+-------------------------
       0    |   0,  1,  3  |  <1,1>, <NULL>,  <NULL>
       1    |   1,  2,  3  | <NULL>,  <0,0>,  <NULL>

   where <Ti,Ei> is a half-edge, where Ti is the *neighbor* triangle index, and
   Ei is the local edge index of Ti that correponds to the half-edge. <NULL> means
   there is no neighbor triangle.

   Vertex-to-Half-Edge Data Struct:

     vertex |  adjacent
    indices | half-edge
   ---------+------------
       0    |   <0,2>
       1    |   <1,2>
       2    |   <1,0>
       3    |   <0,1>

   Diagram depicting half-edges:

                   <1,0>
        V3 +-------------------+ V2
           |\                  |
           |  \          T1    |
           |    \              |
           |      \  <1,1>     |
     <0,1> |        \          | <1,2>
           |    <0,0> \        |
           |            \      |
           |              \    |
           |     T0         \  |
           |                  \|
        V0 +-------------------+ V1
                   <0,2>

   Note: in this example, only need one adjacent half-edge because there are no
         non-manifold vertices.  But we do allow for non-manifold vertices!


    """

    def __init__(self):

        self._reserve_buffer = 0.2
        self.VtxMap = None
        self._size = 0

    def __str__(self):
        OUT_STR = ("The size of the Vertex-to-Half-Facet Map is: " + str(self._size) + "\n"
                 + "The *reserved* size of the Vertex-to-Half-Facet Map is: " + str(len(self.VtxMap)) + "\n" )
        return OUT_STR

    def Clear(self):
        del(self.VtxMap)
        self.VtxMap = None
        self._size = 0

    def Size(self):
        return self._size

    # // return non-const reference to internal data
    # std::vector<VtxHalfFacetType>& Get_VtxMap() { return VtxMap; };

    def Reserve(self, num_VM):
        """This just pre-allocates, or re-sizes.
         The _size attribute is unchanged."""
        # compute the space needed (with extra) to allocate
        Desired_Size = np.rint(np.ceil((1.0 + self._reserve_buffer) * num_VM))
        if self.VtxMap is None:
            # fill the array with null values to start
            self.VtxMap = np.full(Desired_Size.astype(VtxIndType), NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        elif self.VtxMap.size < Desired_Size:
            old_size = self.VtxMap.size
            self.VtxMap = np.resize(self.VtxMap,Desired_Size)
            # put in NULL values
            self.VtxMap[old_size:Desired_Size] = NULL_VtxHalfFacet
        else:
            pass

    def Append(self, *args):
        """Append a (vertex, half-facet) pair; half-facet = (cell index, local facet index)
        if one argument is given, it should be a VtxHalfFacetType;
        elif two arguments are given, it should be a vertex index, followed by a HalfFacetType;
        else three arguments are given, (vtx index, cell index, local facet index).
        """
        
        if (self.VtxMap is None) or (self.VtxMap.size==self._size):
            # need to reserve space
            Reserve(self, self._size+10)
        
        if len(args)==1:
            if args[0].dtype!=VtxHalfFacetType:
                print("Error: input is not a VtxHalfFacetType!")
            self.VtxMap[self._size] = args[0]
            self._size += 1
        elif len(args)==2:
            if args[1].dtype!=HalfFacetType:
                print("Error: second input should be a HalfFacetType!")
            self.VtxMap[self._size] = (args[0], args[1]['ci'], args[1]['fi'])
            self._size += 1
        elif len(args)==3:
            self.VtxMap[self._size] = (args[0], args[1], args[2])
            self._size += 1
        else:
            print("incorrect number of arguments!")

    def Get_Unique_Vertices(self):
        """Get unique list of vertices.
        Does not require 'Sort()' to have been run.
        """
        
        all_vertices = np.zeros(self._size, dtype=VtxIndType)
        for kk in range(self._size):
            # extract out the vertex indices
            all_vertices[kk] = self.VtxMap[kk]['vtx']

        unique_vertices = np.unique(all_vertices)
        return unique_vertices

    def Display_Unique_Vertices(self):
        """Print unique list of vertices to the screen.
        Does not require 'Sort()' to have been run.
        """
        
        unique_vertices = self.Get_Unique_Vertices()
        
        print("Unique list of vertex indices:")
        print(str(unique_vertices[0]), end="")
        for kk in range(1, unique_vertices.size):
            print(", " + str(unique_vertices[kk]), end="")
        print("")

    def Sort(self):
        """Sort the VtxMap so it is useable."""
        self.VtxMap = np.sort(self.VtxMap, order=['vtx', 'ci', 'fi'])

    # Note: all methods below this line require VtxMap to be sorted
    # before they will work correctly.  Make sure to run 'Sort()' first!

    def Get_Half_Facets(self, vi, return_array=""):
        """Find *all* half-facets attached to the given vertex.
        Requires 'Sort()' to have been run to work correctly.
        
        If only the vertex index is given, then returns the first index into the
        VtxMap and the total number of half-facets.
        If a second argument is given as "array", then an array of
        type HalfFacetType is returned instead, that contains the half-facets.
        This second option is mainly for testing.
        """
        #self.VtxMap = np.sort(self.VtxMap, order=['vtx', 'ci', 'fi'])
        first_index       = np.searchsorted(self.VtxMap['vtx'], vi)
        last_index_plus_1 = np.searchsorted(self.VtxMap['vtx'], vi, side='right')
        total = last_index_plus_1 - first_index
        
        if return_array.lower() == "array":
            HFs = np.full(total.astype(VtxIndType), NULL_HalfFacet, dtype=HalfFacetType)
            # copy it over
            for kk in range(total):
                HFs[kk] = self.VtxMap[first_index + kk][['ci','fi']]
            return HFs
        else:
            return first_index, total

    def Get_Half_Facet(self, vi):
        """Return a single half-facet attached to the given vertex.
        Requires 'Sort()' to have been run to work correctly.
        Note: if you only want the index (into VtxMap) of the vertex/half-facet,
        then use 'Get_Half_Facets'.
        """
        first_index, total = self.Get_Half_Facets(vi)
        hf = np.array(self.VtxMap[first_index][['ci','fi']], dtype=HalfFacetType)
        return hf

    def Display_Half_Facets(self, vi=NULL_Vtx):
        """Print out half-facets attached to a given vertex.
        If no argument is given, or NULL_Vtx is given,
        then all half-facets for all stored vertices will be displayed.
        Requires 'Sort()' to have been run to work correctly.
        """
        
        if (vi==NULL_Vtx):
            # we want all the half-facet(s) for all the stored vertices
            Prev_Vtx    = NULL_Vtx
            Current_Vtx = NULL_Vtx
            print("Vertex and attached half-facets, (cell index, local facet index):")
            for kk in range(self._size):
                # get the current vertex
                vhf = self.VtxMap[kk]
                Current_Vtx = vhf['vtx']
                if (Current_Vtx!=Prev_Vtx):
                    # found a new vertex
                    print("")
                    print(str(Current_Vtx) + ": (" + str(vhf['ci']) + ", " + str(vhf['fi']) + ")", end="")
                    Prev_Vtx = Current_Vtx # update
                elif (Current_Vtx==Prev_Vtx):
                    # still with the same vertex, so print the current half-facet
                    print(", ", end="")
                    print("(" + str(vhf['ci']) + ", " + str(vhf['fi']) + ")", end="")
            print("")
        else:
            # print out half-facets for one particular vertex
            first_index, total = self.Get_Half_Facets(vi)
            print("Half-facets (cell index, local facet index) attached to Vtx# " + str(vi) + ":")
            for kk in range(total-1):
                print("(" + str(self.VtxMap[first_index + kk]['ci']) + ", " + \
                            str(self.VtxMap[first_index + kk]['fi']) + "), ", end="")
            # print the last one
            print("(" + str(self.VtxMap[total-1]['ci']) + ", " + \
                            str(self.VtxMap[total-1]['fi']) + ")")
    # 
