"""
ahf.Vtx2HalfFacetMap.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class for storing arrays of mappings from vertex indices to (several)
incident half-facets.

Also, see "BaseMesh.py" for more explanation.

Copyright (c) 12-10-2020,  Shawn W. Walker
"""

#import abc
#from dataclasses import dataclass
#from typing import Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np  # type: ignore
import ahf

# define data types

# half-facet
HalfFacetType = np.dtype({'names': ['ci', 'fi'],
                       'formats': [ahf.CellIndType, ahf.SmallIndType],
                       'titles': ['cell index', 'local facet (entity) index']})
NULL_HalfFacet = np.array((ahf.NULL_Cell, ahf.NULL_Small), dtype=HalfFacetType)

# vertex half-facet
VtxHalfFacetType = np.dtype({'names': ['vtx', 'ci', 'fi'],
                         'formats': [ahf.VtxIndType, ahf.CellIndType, ahf.SmallIndType],
                         'titles': ['global vertex index', 'cell index', 'local facet (entity) index']})
NULL_VtxHalfFacet = np.array((ahf.NULL_Vtx, ahf.NULL_Cell, ahf.NULL_Small), dtype=VtxHalfFacetType)


class Vtx2HalfFacetMap:
    """
    Class for mapping from a given vertex index to (several) incident
    half-facets.  This class stores an array of these mappings.
    Note: this is generic, meaning this can be used for half-facets in
          0-D, 1-D, 2-D, and 3-D meshes (or higher dimensions!).

    EXAMPLE:

      Diagram depicting half-edges (half-facets for 2-D meshes):

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

    Triangle Connectivity:

    triangle |   vertices
    indices |  V0, V1, V2
    ---------+--------------
       0    |   0,  1,  3
       1    |   1,  2,  3

    Half-Edges attached to vertices:

       Vertex V0:  V0--><0,1>
                   V0--><0,2>
       Vertex V1:  V1--><0,2>
                   V1--><0,0>
                   V1--><1,1>
                   V1--><1,2>
       etc...

    where <Ti,Ei> is a half-edge attached to Vi, where Ti (the cell index) and
    Ei (the local edge index) define the particular half-edge.
    """

    def __init__(self):

        self._reserve_buffer = 0.2
        self.VtxMap = None
        self._size = 0

    def __str__(self):
        return "The size of the Vertex-to-Half-Facet Map is: " + str(self._size)

    def Clear(self):
        self.VtxMap = None
        self._size = 0

    def Size(self):
        return self._size

    def Reserve(self, num_VM):
        """This just pre-allocates, or re-sizes.
         The _size attribute is unchanged."""
        # compute the space needed (with extra) to allocate
        Desired_Size = np.ceil((1.0 + self._reserve_buffer) * num_VM);
        if not self.VtxMap:
            self.VtxMap = np.full(Desired_Size, NULL_VtxHalfFacet, dtype=VtxHalfFacetType)
        elif self.VtxMap.size < Desired_Size:
            old_size = self.VtxMap.size
            self.VtxMap = np.resize(self.VtxMap,Desired_Size)
            # put in NULL values
            self.VtxMap[old_size:Desired_Size] = NULL_VtxHalfFacet
        else:
            pass



#
