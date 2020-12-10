"""
ahf
~~~~~~
The ahf package - a Python package for the Array-based Half-facet data
structure.  It is a mesh class.

Copyright (c) 12-10-2020,  Shawn W. Walker
"""

import numpy as np

# define various types

# type for "small" indices (i.e. 0 through ~255)
SmallIndType = np.dtype('uint8');
# type for "medium" indices (i.e. 0 through ~65,535)
MedIndType = np.dtype('uint16');
# vertex indices ("large")
VtxIndType = np.dtype('uint64');
# cell indices ("large"), e.g. triangles, tetrahedrons
CellIndType = np.dtype('uint64');

# define NULL constants
NULL_Small = np.iinfo(SmallIndType).max
NULL_Med   = np.iinfo(MedIndType).max
NULL_Vtx   = np.iinfo(VtxIndType).max
NULL_Cell  = np.iinfo(CellIndType).max
# note: we can never have an index of the max size (should not be a problem)

# define type for "real number"
RealType = np.dtype('float64');

# define type for vertex point coordinates
PointType = np.dtype('float64');

#
