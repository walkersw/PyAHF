"""
============================================================================================
   Class for mapping from a given vertex index to (several) incident half-facets.
   Note: this is generic, meaning this can be used for half-facets in 0-D, 1-D, 2-D,
         and 3-D meshes (or higher dimensions!).

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

   Also, see "BaseMesh.cc" for more explanation.

   Copyright (c) 10-01-2020,  Shawn W. Walker
============================================================================================
"""



