import unittest

import numpy as np
import json

from autolab_core import RigidTransform
from perception import CameraIntrinsics

from make_segmentation_imgs import get_cuboid, get_cuboid2d_visibility


class TestMakeSegmentationImgs(unittest.TestCase):

    def test_get_cuboid(self):
        object_pose = RigidTransform(rotation=np.array([[9.94856600e-01, 9.59313538e-04, 1.01288816e-01],
                                                        [4.85117785e-02, 8.73304839e-01, - 4.84752788e-01],
                                                        [-8.89210435e-02, 4.87173212e-01, 8.68766545e-01]]),
                                     translation=np.array(
                                         [0.023264999389648438, -0.050949001312255859, 1.3201980590820313]),
                                     from_frame='target_model',
                                     to_frame='camera')

        # Qtn: [0.96655677 0.25138875 0.0491978  0.01229945]

        cuboid_dimensions = np.array([0.10240400314331055, 0.140177001953125, 0.10230899810791016])
        camera_intrinsics = CameraIntrinsics(
            frame='camera',
            fx=768.16058349609375,
            fy=768.16058349609375,
            cx=480,
            cy=270,
            skew=0,
            height=540,
            width=960
        )
        expected_json = json.loads(
            """
            {
            "cuboid_centroid": [0.023264999389648438, -0.050949001312255859, 1.3201980590820313],
            "cuboid": [
                          [0.079317998886108398, -0.13447099685668945, 1.32593994140625],
                          [-0.022558999061584473, -0.13944000244140625, 1.3350430297851563],
                          [-0.022427000999450684, -0.017024999856948853, 1.4033380126953125],
                          [0.079450998306274414, -0.012056000232696533, 1.3942340087890625],
                          [0.068956999778747559, -0.084874000549316406, 1.2370590209960938],
                          [-0.032920000553131104, -0.089841995239257813, 1.2461630249023438],
                          [-0.032788000106811523, 0.032572999000549316, 1.3144569396972656],
                          [0.069088997840881348, 0.037541000843048096, 1.3053529357910156]
                      ],
            "projected_cuboid_centroid": [493.53689575195313, 240.35519409179688],
            "projected_cuboid": [
                [525.9517822265625, 192.09669494628906],
                [467.01998901367188, 189.76899719238281],
                [467.72409057617188, 260.68121337890625],
                [523.77362060546875, 263.3577880859375],
                [522.81939697265625, 217.29719543457031],
                [459.70730590820313, 214.6195068359375],
                [460.8389892578125, 289.035400390625],
                [520.6571044921875, 292.09201049804688]
            ]
            }
            """
        )
        actual_json = get_cuboid(object_pose, cuboid_dimensions, camera_intrinsics)

        np.testing.assert_almost_equal(actual_json['cuboid_centroid'],
                                       expected_json['cuboid_centroid'], decimal=4)
        np.testing.assert_almost_equal(actual_json['cuboid'],
                                       expected_json['cuboid'], decimal=4)
        np.testing.assert_almost_equal(actual_json['projected_cuboid_centroid'],
                                       expected_json['projected_cuboid_centroid'], decimal=2)
        np.testing.assert_almost_equal(actual_json['projected_cuboid'],
                                       expected_json['projected_cuboid'], decimal=2)

    def test_get_cuboid2d_visibility_corner(self):
        img_width = 640
        img_height = 480
        cuboid2d = [[-100.0, -100.0], [100.0, 100.0], [100.0, -100.0], [-100.0, 100.0],
                    [1, 2], [3, 4], [5, 6], [7, 8]]
        vis = get_cuboid2d_visibility(cuboid2d, img_width, img_height)
        self.assertAlmostEqual(vis, 0.25)

    def test_get_cuboid2d_visibility_edge(self):
        img_width = 640
        img_height = 480
        cuboid2d = [[100.0, -100.0], [200.0, -100.0], [200.0, 100.0], [100.0, 100.0]]
        vis = get_cuboid2d_visibility(cuboid2d, img_width, img_height)
        self.assertAlmostEqual(vis, 0.5)

    def test_get_cuboid2d_visibility_full(self):
        img_width = 640
        img_height = 480
        cuboid2d = [[0, 10], [20, 40], [70, 15], [20, 20]]
        vis = get_cuboid2d_visibility(cuboid2d, img_width, img_height)
        self.assertAlmostEqual(vis, 1.0)

    def test_get_cuboid2d_visibility_zero(self):
        img_width = 640
        img_height = 480
        cuboid2d = [[0, 0], [0, 40], [-70, -15]]
        vis = get_cuboid2d_visibility(cuboid2d, img_width, img_height)
        self.assertAlmostEqual(vis, 0.0)



if __name__ == '__main__':
    unittest.main()
