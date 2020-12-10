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

    def __init__(self)

        if data_price_ord[0]['price'] == 0:
            raise ValueError('invalid price of 0 in supply data')

        for n, e in enumerate(data_price_ord[1:]):
            current_point = e['supply']
            previous_point = data_price_ord[n]['supply']
            if current_point < previous_point:
                raise SupplyMonotonicityError

        self._price = np.array([d['price'] for d in data_price_ord])
        self._quantity = np.array([d['supply'] for d in data_price_ord])
        self._min_price = self._price.min()

    def __eq__(self, other):
        

    def quantity(self, price: float):
        """Return supply quantity for a given price.
        :param price: Price.
        :type price: float
        :return: Quantity.
        :rtype: float
        """

        if price < self._min_price:
            quantity_at_price = 0.
        else:
            quantity_at_price = self._quantity[self._price <= price][-1]
        return quantity_at_price


#
