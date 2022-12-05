import unittest

import numpy as np
#from ahf import *
#import ahf as AHF

from ahf.SimplexMath import *

class TestSimplexMath(unittest.TestCase):

    # @classmethod
    # def setUpClass(self):

    # @classmethod
    # def tearDownClass(self):

    # def setUp(self):

    def tearDown(self):
        print(" ")

    def test_Affine_Map(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])

        print(vc0)
        A0, b0 = Affine_Map(vc0)
        print(A0)
        print(b0)
        J_sq = np.matmul(A0.T,A0)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 10.0, "Should be 10.0.")
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])

        print(vc1)
        A1, b1 = Affine_Map(vc1)
        print(A1)
        print(b1)
        J_sq = np.matmul(A1.T,A1)
        J_sq_det = np.linalg.det(J_sq)
        J_det = np.round(np.sqrt(J_sq_det), 8)
        print(J_det)
        self.assertEqual(J_det, 11.67155088, "Should be 11.67155088.")

        vc2 = np.array([vc0, vc1])
        A2, b2 = Affine_Map(vc2)
        print(A2)
        print(b2)
        A2_CHK = np.array([A0, A1])
        b2_CHK = np.array([b0, b1])
        self.assertEqual(np.array_equal(A2_CHK,A2), True, "Should be True.")
        self.assertEqual(np.array_equal(b2_CHK,b2), True, "Should be True.")

    def test_Ref2Cart_and_back(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        rc0 = np.array([0.3, 0.2])
        print(rc0)

        cart0 = Reference_To_Cartesian(vc0, rc0)
        print(cart0)
        diff0 = np.amax(np.abs(cart0 - np.array([0.6, 1.9, 1.1], dtype=CoordType)))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        rc0_CHK = Cartesian_To_Reference(vc0, cart0)
        diff_rc0 = np.amax(np.abs(rc0 - rc0_CHK))
        self.assertEqual(diff_rc0 < 1e-15, True, "Should be True.")
        #print(rc0_CHK)
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        rc1 = np.array([0.1, 0.7])
        print(rc1)

        cart1 = Reference_To_Cartesian(vc1, rc1)
        print(cart1)
        diff1 = np.amax(np.abs(cart1 - np.array([2.33, 0.79, 3.12], dtype=CoordType)))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        rc1_CHK = Cartesian_To_Reference(vc1, cart1)
        diff_rc1 = np.amax(np.abs(rc1 - rc1_CHK))
        self.assertEqual(diff_rc1 < 1e-15, True, "Should be True.")
        #print(rc1_CHK)

        vc2 = np.array([vc0, vc1])
        rc2 = np.array([rc0, rc1])
        cart2 = Reference_To_Cartesian(vc2, rc2)
        #print(cart2)
        cart2_CHK = np.array([cart0, cart1])
        self.assertEqual(np.array_equal(cart2_CHK,cart2), True, "Should be True.")

        rc2_CHK = Cartesian_To_Reference(vc2, cart2)
        diff_rc2 = np.amax(np.abs(rc2 - rc2_CHK))
        self.assertEqual(diff_rc2 < 1e-15, True, "Should be True.")

    def test_Bary2Ref_and_back(self):
        bc0 = np.array([0.5, 0.3, 0.2])
        print(bc0)
        rc0 = Barycentric_To_Reference(bc0)
        print(rc0)
        self.assertEqual(np.array_equal(bc0[1:],rc0), True, "Should be True.")
        bc0_CHK = Reference_To_Barycentric(rc0)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")

        bc1 = np.array([0.2, 0.1, 0.7])
        print(bc1)
        rc1 = Barycentric_To_Reference(bc1)
        print(rc1)
        self.assertEqual(np.array_equal(bc1[1:],rc1), True, "Should be True.")
        bc1_CHK = Reference_To_Barycentric(rc1)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")

        bc2 = np.array([bc0, bc1])
        rc2 = Barycentric_To_Reference(bc2)
        #print(rc2)
        rc2_CHK = np.array([rc0, rc1])
        diff_rc2 = np.amax(np.abs(rc2 - rc2_CHK))
        self.assertEqual(diff_rc2 < 1e-15, True, "Should be True.")
        bc2_CHK = Reference_To_Barycentric(rc2)
        diff_bc2 = np.amax(np.abs(bc2 - bc2_CHK))
        self.assertEqual(diff_bc2 < 1e-15, True, "Should be True.")

    def test_Bary2Cart_and_back(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        bc0 = np.array([0.5, 0.3, 0.2])
        print(bc0)

        cart0 = Barycentric_To_Cartesian(vc0, bc0)
        print(cart0)
        diff0 = np.amax(np.abs(cart0 - np.array([0.6, 1.9, 1.1], dtype=CoordType)))
        self.assertEqual(diff0 < 1e-15, True, "Should be True.")
        bc0_CHK = Cartesian_To_Barycentric(vc0, cart0)
        diff_bc0 = np.amax(np.abs(bc0 - bc0_CHK))
        self.assertEqual(diff_bc0 < 1e-15, True, "Should be True.")
        #print(bc0_CHK)
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        bc1 = np.array([0.2, 0.1, 0.7])
        print(bc1)

        cart1 = Barycentric_To_Cartesian(vc1, bc1)
        print(cart1)
        diff1 = np.amax(np.abs(cart1 - np.array([2.33, 0.79, 3.12], dtype=CoordType)))
        self.assertEqual(diff1 < 1e-15, True, "Should be True.")
        bc1_CHK = Cartesian_To_Barycentric(vc1, cart1)
        diff_bc1 = np.amax(np.abs(bc1 - bc1_CHK))
        self.assertEqual(diff_bc1 < 1e-15, True, "Should be True.")
        #print(bc1_CHK)

        vc2 = np.array([vc0, vc1])
        bc2 = np.array([bc0, bc1])
        cart2 = Barycentric_To_Cartesian(vc2, bc2)
        #print(cart2)
        cart2_CHK = np.array([cart0, cart1])
        self.assertEqual(np.array_equal(cart2_CHK,cart2), True, "Should be True.")

        bc2_CHK = Cartesian_To_Barycentric(vc2, cart2)
        diff_bc2 = np.amax(np.abs(bc2 - bc2_CHK))
        self.assertEqual(diff_bc2 < 1e-15, True, "Should be True.")

    def test_Measures(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        D0 = Diameter(vc0)
        self.assertEqual(np.abs(D0 - 5.0) < 1e-15, True, "Should be True.")
        BB_min0, BB_max0 = Bounding_Box(vc0)
        self.assertEqual(np.array_equal(BB_min0,np.array([0.0, 0.0, 1.1])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max0,np.array([2.0, 5.0, 1.1])), True, "Should be True.")
        V0 = Volume(vc0)
        #print(V0)
        self.assertEqual(np.abs(V0 - 5.0) < 1e-15, True, "Should be True.")
        P0, Facet_V0 = Perimeter(vc0)
        Facet_V0_CHK = np.array([np.sqrt(8), 5.0, np.sqrt(13)])
        diff_Facet_V0 = np.amax(np.abs(Facet_V0 - Facet_V0_CHK))
        self.assertEqual(diff_Facet_V0 < 1e-15, True, "Should be True.")
        self.assertEqual(np.abs(P0 - np.sum(Facet_V0_CHK)) < 1e-15, True, "Should be True.")
        
        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        D1 = Diameter(vc1)
        self.assertEqual(np.abs(D1 - 5.707889277132134) < 1e-15, True, "Should be True.")
        BB_min1, BB_max1 = Bounding_Box(vc1)
        self.assertEqual(np.array_equal(BB_min1,np.array([0.2, 0.3, -0.4])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max1,np.array([3.1, 2.4, 4.5])), True, "Should be True.")
        V1 = Volume(vc1)
        #print(V1)
        self.assertEqual(np.abs(V1 - 5.835775441190314) < 1e-15, True, "Should be True.")
        P1, Facet_V1 = Perimeter(vc1)
        Facet_V1_CHK = np.array([4.743416490252569, 5.707889277132135, 2.493992782667986])
        diff_Facet_V1 = np.amax(np.abs(Facet_V1 - Facet_V1_CHK))
        self.assertEqual(diff_Facet_V1 < 1e-15, True, "Should be True.")
        self.assertEqual(np.abs(P1 - np.sum(Facet_V1_CHK)) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        D2 = Diameter(vc2)
        D2_CHK = np.array([D0, D1])
        diff_D2 = np.amax(np.abs(D2 - D2_CHK))
        self.assertEqual(diff_D2 < 1e-15, True, "Should be True.")
        BB_min2, BB_max2 = Bounding_Box(vc2)
        self.assertEqual(np.array_equal(BB_min2,np.array([BB_min0, BB_min1])), True, "Should be True.")
        self.assertEqual(np.array_equal(BB_max2,np.array([BB_max0, BB_max1])), True, "Should be True.")
        V2 = Volume(vc2)
        V2_CHK = np.array([V0, V1])
        diff_V2 = np.amax(np.abs(V2 - V2_CHK))
        self.assertEqual(diff_V2 < 1e-15, True, "Should be True.")
        P2, Facet_V2 = Perimeter(vc2)
        Facet_V2_CHK = np.array([[np.sqrt(8), 5.0, np.sqrt(13)], \
                                [4.743416490252569, 5.707889277132135, 2.493992782667986]])
        diff_Facet_V2 = np.amax(np.abs(Facet_V2 - Facet_V2_CHK))
        self.assertEqual(diff_Facet_V2 < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(P2 - np.sum(Facet_V2_CHK, axis=1))) < 1e-15, True, "Should be True.")

    def test_Frames(self):
        # test full orthogonal frames
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        Ortho_0 = Orthogonal_Frame(vc0)
        print(Ortho_0)
        normal_0_CHK = np.array([[0.0], [0.0], [1.0]])
        self.assertEqual(np.amax(np.abs(Ortho_0[:,[-1]] - normal_0_CHK)) < 1e-15, True, "Should be True.")

        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        Ortho_1 = Orthogonal_Frame(vc1)
        print(Ortho_1)
        A1, b1 = Affine_Map(vc1)
        normal_1_CHK = np.cross(A1[:,0], A1[:,1])
        normal_1_CHK = normal_1_CHK.reshape((3,1))
        normal_1_CHK = normal_1_CHK * (1.0/np.linalg.norm(normal_1_CHK,2))
        self.assertEqual(np.amax(np.abs(Ortho_1[:,[-1]] - normal_1_CHK)) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        print(vc2)
        Ortho_2 = Orthogonal_Frame(vc2)
        print(Ortho_2)
        normal_2_CHK = np.array([normal_0_CHK, normal_1_CHK])
        self.assertEqual(np.amax(np.abs(Ortho_2[:,:,[-1]] - normal_2_CHK)) < 1e-15, True, "Should be True.")

        # test full orthogonal frames (1-D line in 2-D)
        pc0 = np.array([[0.2, 0.1], [0.8, 0.6]])
        print(pc0)
        Ortho_pc0 = Orthogonal_Frame(pc0)
        #print(Ortho_pc0)
        tan_pc0_CHK = pc0[1,:] - pc0[0,:]
        tan_pc0_CHK = tan_pc0_CHK.reshape((2,1))
        tan_pc0_CHK = tan_pc0_CHK * (1.0/np.linalg.norm(tan_pc0_CHK,2))
        nor_pc0_CHK = np.array([-tan_pc0_CHK[1], tan_pc0_CHK[0]])
        Ortho_pc0_CHK = np.hstack((tan_pc0_CHK, nor_pc0_CHK))
        self.assertEqual(np.amax(np.abs(Ortho_pc0 - Ortho_pc0_CHK)) < 1e-15, True, "Should be True.")

        pc1 = np.array([[-0.3, 0.3], [0.1, 1.1]])
        print(pc1)
        Ortho_pc1 = Orthogonal_Frame(pc1)
        #print(Ortho_pc1)
        tan_pc1_CHK = pc1[1,:] - pc1[0,:]
        tan_pc1_CHK = tan_pc1_CHK.reshape((2,1))
        tan_pc1_CHK = tan_pc1_CHK * (1.0/np.linalg.norm(tan_pc1_CHK,2))
        nor_pc1_CHK = np.array([-tan_pc1_CHK[1], tan_pc1_CHK[0]])
        Ortho_pc1_CHK = np.hstack((tan_pc1_CHK, nor_pc1_CHK))
        self.assertEqual(np.amax(np.abs(Ortho_pc1 - Ortho_pc1_CHK)) < 1e-15, True, "Should be True.")

        pc2 = np.array([pc0, pc1])
        print(pc2)
        Ortho_pc2 = Orthogonal_Frame(pc2)
        #print(Ortho_pc2)
        Ortho_pc2_CHK = np.array([Ortho_pc0_CHK, Ortho_pc1_CHK])
        self.assertEqual(np.amax(np.abs(Ortho_pc2 - Ortho_pc2_CHK)) < 1e-15, True, "Should be True.")

        # test the tangent space
        TS_0 = Tangent_Space(vc0)
        print(TS_0)
        # build normal space projection
        NS_0_proj = np.identity(3) - (np.outer(TS_0[:,0],TS_0[:,0]) + np.outer(TS_0[:,1],TS_0[:,1]))
        NS_0_CHK = np.outer(normal_0_CHK[:,0],normal_0_CHK[:,0])
        self.assertEqual(np.amax(np.abs(NS_0_proj - NS_0_CHK)) < 1e-15, True, "Should be True.")
        
        TS_1 = Tangent_Space(vc1)
        print(TS_1)
        # build normal space projection
        NS_1_proj = np.identity(3) - (np.outer(TS_1[:,0],TS_1[:,0]) + np.outer(TS_1[:,1],TS_1[:,1]))
        NS_1_CHK = np.outer(normal_1_CHK[:,0],normal_1_CHK[:,0])
        self.assertEqual(np.amax(np.abs(NS_1_proj - NS_1_CHK)) < 1e-15, True, "Should be True.")
        
        TS_2 = Tangent_Space(vc2)
        print(TS_2)
        # build normal space projection
        NS_2_proj = np.array([0*NS_0_proj, 0*NS_1_proj])
        NS_2_proj[0,:,:] = np.identity(3) - (np.outer(TS_2[0,:,0],TS_2[0,:,0]) + np.outer(TS_2[0,:,1],TS_2[0,:,1]))
        NS_2_proj[1,:,:] = np.identity(3) - (np.outer(TS_2[1,:,0],TS_2[1,:,0]) + np.outer(TS_2[1,:,1],TS_2[1,:,1]))
        NS_2_CHK = np.array([NS_0_CHK, NS_1_CHK])
        self.assertEqual(np.amax(np.abs(NS_2_proj - NS_2_CHK)) < 1e-15, True, "Should be True.")

        # test tangent space (1-D line in 2-D)
        TS_pc0 = Tangent_Space(pc0)
        self.assertEqual(np.amax(np.abs(TS_pc0 - tan_pc0_CHK)) < 1e-15, True, "Should be True.")

        TS_pc1 = Tangent_Space(pc1)
        self.assertEqual(np.amax(np.abs(TS_pc1 - tan_pc1_CHK)) < 1e-15, True, "Should be True.")

        TS_pc2 = Tangent_Space(pc2)
        tan_pc2_CHK = np.array([tan_pc0_CHK, tan_pc1_CHK])
        self.assertEqual(np.amax(np.abs(TS_pc2 - tan_pc2_CHK)) < 1e-15, True, "Should be True.")

        # test the normal space
        NS_0 = Normal_Space(vc0)
        print(NS_0)
        # build normal space projection
        NS_0_proj_alt = np.outer(NS_0[:,0],NS_0[:,0])
        self.assertEqual(np.amax(np.abs(NS_0_proj_alt - NS_0_CHK)) < 1e-15, True, "Should be True.")
        
        NS_1 = Normal_Space(vc1)
        print(NS_1)
        # build normal space projection
        NS_1_proj_alt = np.outer(NS_1[:,0],NS_1[:,0])
        self.assertEqual(np.amax(np.abs(NS_1_proj_alt - NS_1_CHK)) < 1e-15, True, "Should be True.")
        
        NS_2 = Normal_Space(vc2)
        print(NS_2)
        # build normal space projection
        NS_2_proj_alt = np.array([0*NS_0_proj_alt, 0*NS_1_proj_alt])
        NS_2_proj_alt[0,:,:] = np.outer(NS_2[0,:,0],NS_2[0,:,0])
        NS_2_proj_alt[1,:,:] = np.outer(NS_2[1,:,0],NS_2[1,:,0])
        self.assertEqual(np.amax(np.abs(NS_2_proj_alt - NS_2_CHK)) < 1e-15, True, "Should be True.")

        # test normal space (1-D line in 2-D)
        NS_pc0 = Normal_Space(pc0)
        self.assertEqual(np.amax(np.abs(NS_pc0 - nor_pc0_CHK)) < 1e-15, True, "Should be True.")

        NS_pc1 = Normal_Space(pc1)
        self.assertEqual(np.amax(np.abs(NS_pc1 - nor_pc1_CHK)) < 1e-15, True, "Should be True.")

        NS_pc2 = Normal_Space(pc2)
        nor_pc2_CHK = np.array([nor_pc0_CHK, nor_pc1_CHK])
        self.assertEqual(np.amax(np.abs(NS_pc2 - nor_pc2_CHK)) < 1e-15, True, "Should be True.")

    def test_Hyperplane_Closest_Point(self):
        # test hyperplane/closest point methods
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        pY0 = np.array([[-8.2], [5.6], [-2.3]])
        print(vc0)
        print(pY0)
        X_star_0, Diff_Vec_0 = Hyperplane_Closest_Point(vc0,pY0)
        print(X_star_0)
        print(Diff_Vec_0)
        X_star_0_CHK = np.array([[-8.2], [5.6], [1.1]])
        DV_0_CHK = pY0 - X_star_0_CHK
        self.assertEqual(np.amax(np.abs(X_star_0 - X_star_0_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(Diff_Vec_0 - DV_0_CHK)) < 1e-15, True, "Should be True.")

        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        pY1 = np.array([[-3.7], [4.9], [5.1]])
        print(vc1)
        print(pY1)
        X_star_1, Diff_Vec_1 = Hyperplane_Closest_Point(vc1,pY1)
        print(X_star_1)
        print(Diff_Vec_1)
        A1, b1 = Affine_Map(vc1)
        normal_1_CHK = np.cross(A1[:,0], A1[:,1])
        normal_1_CHK = normal_1_CHK.reshape((3,1))
        normal_1_CHK = normal_1_CHK * (1.0/np.linalg.norm(normal_1_CHK,2))
        # we can compute the difference w.r.t. any of the simplex vertices
        DV_1_CHK = np.dot((pY1[:,0] - vc1[1,:]),normal_1_CHK) * normal_1_CHK
        X_star_1_CHK = pY1 - DV_1_CHK
        #print(DV_1_CHK)
        self.assertEqual(np.amax(np.abs(X_star_1 - X_star_1_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(Diff_Vec_1 - DV_1_CHK)) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        pY2 = np.array([pY0, pY1])
        print(vc2)
        print(pY2)
        X_star_2, Diff_Vec_2 = Hyperplane_Closest_Point(vc2,pY2)
        print(X_star_2)
        print(Diff_Vec_2)
        X_star_2_CHK = np.array([X_star_0_CHK, X_star_1_CHK])
        DV_2_CHK = np.array([DV_0_CHK, DV_1_CHK])
        #print(DV_2_CHK)
        self.assertEqual(np.amax(np.abs(X_star_2 - X_star_2_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(Diff_Vec_2 - DV_2_CHK)) < 1e-15, True, "Should be True.")

    def test_Centers(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        bc0 = Barycenter(vc0)
        #print(bc0)
        bc0_CHK = np.array([(2.0/3), (8.0/3), 1.1])
        self.assertEqual(np.amax(np.abs(bc0 - bc0_CHK)) < 1e-15, True, "Should be True.")

        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        bc1 = Barycenter(vc1)
        #print(bc1)
        bc1_CHK = np.array([(4.5/3), (3.4/3), (4.6/3)])
        self.assertEqual(np.amax(np.abs(bc1 - bc1_CHK)) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        print(vc2)
        bc2 = Barycenter(vc2)
        #print(bc2)
        bc2_CHK = np.array([bc0_CHK, bc1_CHK])
        self.assertEqual(np.amax(np.abs(bc2 - bc2_CHK)) < 1e-15, True, "Should be True.")

        CB0, CR0 = Circumcenter(vc0)
        #print(CB0)
        #print(CR0)
        CB0_CHK = np.array([0.6, -0.25, 0.65])
        CR0_CHK = 2.5495097567963922
        self.assertEqual(np.amax(np.abs(CB0 - CB0_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(CR0 - CR0_CHK)) < 1e-15, True, "Should be True.")

        CB1, CR1 = Circumcenter(vc1)
        #print(CB1)
        #print(CR1)
        CB1_CHK = np.array([0.6730587828527932, -0.2307922695597216, 0.5577334867069286])
        CR1_CHK = 2.8927002160827406
        self.assertEqual(np.amax(np.abs(CB1 - CB1_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(CR1 - CR1_CHK)) < 1e-15, True, "Should be True.")

        CB2, CR2 = Circumcenter(vc2)
        #print(CB2)
        #print(CR2)
        CB2_CHK = np.array([CB0_CHK, CB1_CHK])
        CR2_CHK = np.array([CR0_CHK, CR1_CHK])
        self.assertEqual(np.amax(np.abs(CB2 - CB2_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(CR2 - CR2_CHK)) < 1e-15, True, "Should be True.")

        IB0, IR0 = Incenter(vc0)
        #print(IB0)
        #print(IR0)
        IB0_CHK = np.array([0.2473703400291712, 0.4372931122476224, 0.3153365477232065])
        IR0_CHK = 0.8745862244952449
        self.assertEqual(np.amax(np.abs(IB0 - IB0_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(IR0 - IR0_CHK)) < 1e-15, True, "Should be True.")

        IB1, IR1 = Incenter(vc1)
        #print(IB1)
        #print(IR1)
        IB1_CHK = np.array([0.3664200151052727, 0.4409237264836121, 0.1926562584111152])
        IR1_CHK = 0.9016053849397796
        self.assertEqual(np.amax(np.abs(IB1 - IB1_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(IR1 - IR1_CHK)) < 1e-15, True, "Should be True.")

        IB2, IR2 = Incenter(vc2)
        #print(IB2)
        #print(IR2)
        IB2_CHK = np.array([IB0_CHK, IB1_CHK])
        IR2_CHK = np.array([IR0_CHK, IR1_CHK])
        self.assertEqual(np.amax(np.abs(IB2 - IB2_CHK)) < 1e-15, True, "Should be True.")
        self.assertEqual(np.amax(np.abs(IR2 - IR2_CHK)) < 1e-15, True, "Should be True.")

        # test shape regularity
        SR0 = Shape_Regularity(vc0)
        self.assertEqual(np.abs(SR0 - (CR0/IR0)) < 1e-15, True, "Should be True.")
        SR1 = Shape_Regularity(vc1)
        self.assertEqual(np.abs(SR1 - (CR1/IR1)) < 1e-15, True, "Should be True.")
        SR2 = Shape_Regularity(vc2)
        self.assertEqual(np.amax(np.abs(SR2 - (CR2/IR2))) < 1e-15, True, "Should be True.")

    def test_Angles(self):
        vc0 = np.array([[0.0, 0.0, 1.1], [2.0, 3.0, 1.1], [0.0, 5.0, 1.1]])
        print(vc0)
        Ang0 = Angles(vc0)
        #print(Ang0)
        Ang0_CHK = np.array([0.7853981633974483, 1.768191886644777, 0.5880026035475675])
        self.assertEqual(np.amax(np.abs(Ang0 - Ang0_CHK)) < 1e-15, True, "Should be True.")

        vc1 = np.array([[0.2, 0.3, -0.4], [1.2, 2.4, 0.5], [3.1, 0.7, 4.5]])
        print(vc1)
        Ang1 = Angles(vc1)
        #print(Ang1)
        Ang1_CHK = np.array([0.4456936423253770, 1.734672745776647, 0.9612262654877698])
        self.assertEqual(np.amax(np.abs(Ang1 - Ang1_CHK)) < 1e-15, True, "Should be True.")

        vc2 = np.array([vc0, vc1])
        print(vc2)
        Ang2 = Angles(vc2)
        #print(Ang2)
        Ang2_CHK = np.array([Ang0_CHK, Ang1_CHK])
        self.assertEqual(np.amax(np.abs(Ang2 - Ang2_CHK)) < 1e-15, True, "Should be True.")

        # test angles in a tetrahedron
        tc0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        print(tc0)
        TA0 = Angles(tc0)
        #print(TA0)
        TA0_CHK = np.array([0.9553166181245093, 0.9553166181245093, 0.9553166181245093, \
                            1.570796326794897, 1.570796326794897, 1.570796326794897])
        self.assertEqual(np.amax(np.abs(TA0 - TA0_CHK)) < 1e-15, True, "Should be True.")

        # equilateral tetrahedron
        tc1 = np.array([[np.sqrt(8/9), 0, -(1/3)], [-np.sqrt(2/9), np.sqrt(2/3), -(1/3)], \
                        [-np.sqrt(2/9), -np.sqrt(2/3), -(1/3)], [0, 0, 1]])
        print(tc1)
        TA1 = Angles(tc1)
        #print(TA1)
        TA1_CHK = 1.230959417340775 * np.ones((6,))
        self.assertEqual(np.amax(np.abs(TA1 - TA1_CHK)) < 1e-15, True, "Should be True.")

        tc2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3)/2, 0.0], [0.5, np.sqrt(3)/4, 3/4]])
        print(tc2)
        TA2 = Angles(tc2)
        #print(TA2)
        TA2_CHK = np.array([1.176005207095135, 1.289761425292083, 1.289761425292083, \
                            1.289761425292083, 1.289761425292083, 1.047197551196598])
        self.assertEqual(np.amax(np.abs(TA2 - TA2_CHK)) < 1e-15, True, "Should be True.")

        tc3 = np.array([tc0, tc1, tc2])
        print(tc3)
        TA3 = Angles(tc3)
        #print(TA3)
        TA3_CHK = np.array([TA0_CHK, TA1_CHK, TA2_CHK])
        self.assertEqual(np.amax(np.abs(TA3 - TA3_CHK)) < 1e-15, True, "Should be True.")




if __name__ == '__main__':
    unittest.main()
