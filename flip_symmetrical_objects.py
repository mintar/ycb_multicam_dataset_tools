#!/usr/bin/env python
import argparse
import collections
import copy
import json
import os
import re
import sys

import numpy as np
from autolab_core import RigidTransform, PointCloud, Point
from perception import CameraIntrinsics
from shapely.geometry import MultiPoint

ModelConfig = collections.namedtuple('ModelConfig',
                                     'class_name segmentation_class_id model_transform cuboid_dimensions')


def main():
    # ====================================
    # parse arguments
    # ====================================
    parser = argparse.ArgumentParser(description='Generate segmentation images and updated json files.')
    parser.add_argument('-d', '--data-dir', default=os.getcwd(), help='directory containing images and json files')
    parser.add_argument('-o', '--object-settings',
                        help='object_settings file where the fixed_model_transform corresponds to the poses'
                             'in the frame annotation jsons.'
                             'default: <data_dir>/_object_settings.json')
    parser.add_argument('-t', '--target-object-settings',
                        help='if given, transform all poses into the fixed_model_transform from this file.')
    args = parser.parse_args()

    if not args.object_settings:
        args.object_settings = os.path.join(args.data_dir, '_object_settings.json')
    if not args.target_object_settings:
        args.target_object_settings = args.object_settings

    # =====================
    # parse object_settings
    # =====================
    with open(args.object_settings, 'r') as f:
        object_settings_json = json.load(f)
    with open(args.target_object_settings, 'r') as f:
        target_object_settings_json = json.load(f)

    if not len(object_settings_json['exported_objects']) == len(target_object_settings_json['exported_objects']):
        print "FATAL: object_settings and target_object_settings do not match!"
        sys.exit(-1)
    models = {}
    for model_json, target_model_json in zip(object_settings_json['exported_objects'],
                                             target_object_settings_json['exported_objects']):
        class_name = model_json['class']
        if not class_name == target_model_json['class']:
            print "FATAL: object_settings and target_object_settings do not match!"
            sys.exit(-1)
        segmentation_class_id = model_json['segmentation_class_id']
        cuboid_dimensions = np.array(model_json['cuboid_dimensions'])

        # calculate model_transform
        fixed_model_transform_mat = np.transpose(np.array(model_json['fixed_model_transform']))
        fixed_model_transform = RigidTransform(
            rotation=fixed_model_transform_mat[:3, :3],
            translation=fixed_model_transform_mat[:3, 3],
            from_frame='ycb_model',
            to_frame='source_model'
        )
        target_fixed_model_transform_mat = np.transpose(np.array(target_model_json['fixed_model_transform']))
        target_fixed_model_transform = RigidTransform(
            rotation=target_fixed_model_transform_mat[:3, :3],
            translation=target_fixed_model_transform_mat[:3, 3],
            from_frame='ycb_model',
            to_frame='target_model'
        )
        model_transform = fixed_model_transform.dot(target_fixed_model_transform.inverse())

        models[class_name] = ModelConfig(class_name, segmentation_class_id, model_transform, cuboid_dimensions)

    # ==================================================
    # parse camera_settings and set up camera intrinsics
    # ==================================================
    with open(os.path.join(args.data_dir, '_camera_settings.json'), 'r') as f:
        camera_settings_json = json.load(f)['camera_settings'][0]

    camera_intrinsics = CameraIntrinsics(
        frame='camera',
        fx=camera_settings_json['intrinsic_settings']['fx'],
        fy=camera_settings_json['intrinsic_settings']['fy'],
        cx=camera_settings_json['intrinsic_settings']['cx'],
        cy=camera_settings_json['intrinsic_settings']['cy'],
        skew=camera_settings_json['intrinsic_settings']['s'],
        height=camera_settings_json['captured_image_size']['height'],
        width=camera_settings_json['captured_image_size']['width']
    )

    # ====================================
    # process each frame
    # ====================================
    pattern = re.compile(r'\d{3,}.json')
    json_files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if pattern.match(f)])
    for json_file in json_files:
        filename_prefix = json_file[:-len('json')]
        print '\n---------------------- {}*'.format(filename_prefix)
        with open(json_file, 'r') as f:
            frame_json = json.load(f)

        updated_frame_json = process_frame(frame_json, camera_intrinsics, models)
        with open(filename_prefix + 'flipped.json', 'w') as f:
            json.dump(updated_frame_json, f, indent=2, sort_keys=True)


def process_frame(frame_json, camera_intrinsics, models):
    num_objects = len(frame_json['objects'])
    if len(frame_json['objects']) == 0:
        print "no objects in frame!"
        return frame_json

    # copy relevant fields of json
    updated_frame_json = {'camera_data': {}, 'objects': []}
    if frame_json['camera_data']:
        updated_frame_json['camera_data']['location_worldframe'] = copy.deepcopy(
            frame_json['camera_data']['location_worldframe'])
        updated_frame_json['camera_data']['quaternion_xyzw_worldframe'] = copy.deepcopy(
            frame_json['camera_data']['quaternion_xyzw_worldframe'])
    else:
        print 'Warning: no `camera_data` field!'

    for i in range(len(frame_json['objects'])):
        updated_frame_json['objects'].append({})
        updated_frame_json['objects'][i]['class'] = copy.deepcopy(frame_json['objects'][i]['class'])
        updated_frame_json['objects'][i]['bounding_box'] = copy.deepcopy(frame_json['objects'][i]['bounding_box'])
        updated_frame_json['objects'][i]['instance_id'] = copy.deepcopy(frame_json['objects'][i]['instance_id'])
        updated_frame_json['objects'][i]['visibility'] = copy.deepcopy(frame_json['objects'][i]['visibility'])

    # get object poses and flip symmetrical objects if necessary
    object_poses = []
    for object_index in range(num_objects):
        model_name = updated_frame_json['objects'][object_index]['class']
        scene_object_json = frame_json['objects'][object_index]
        (translation, rotation) = object_pose_from_json(scene_object_json)
        object_pose = RigidTransform(
            rotation=rotation,
            translation=translation,
            from_frame='source_model',
            to_frame='camera'
        )
        object_pose = object_pose.dot(models[model_name].model_transform)
        # TODO: flip
        object_poses.append(object_pose)

        updated_frame_json['objects'][object_index]['location'] = object_pose.translation.tolist()
        updated_frame_json['objects'][object_index]['quaternion_xyzw'] = np.roll(object_pose.quaternion, -1).tolist()

    # add pose_transform_permuted
    for object_index in range(num_objects):
        ptp_json = pose_transform_permuted_to_json(object_poses[object_index].translation,
                                                   object_poses[object_index].rotation)
        updated_frame_json['objects'][object_index].update(ptp_json)

    # compute cuboid, cuboid_centroid, projected_cuboid, projected_cuboid_centroid, bounding_box
    for object_index in range(num_objects):
        model_name = updated_frame_json['objects'][object_index]['class']
        cuboid_json = get_cuboid(object_poses[object_index], models[model_name].cuboid_dimensions, camera_intrinsics)
        updated_frame_json['objects'][object_index].update(cuboid_json)

    return updated_frame_json


def get_cuboid(object_pose, cuboid_dimensions, camera_intrinsics):
    # cuboid centroid
    cuboid_centroid = object_pose.translation
    result_json = {'cuboid_centroid': cuboid_centroid.tolist()}

    # cuboid
    x = cuboid_dimensions[0] / 2
    y = cuboid_dimensions[1] / 2
    z = cuboid_dimensions[2] / 2
    # colors in nvdu_viz:       b   b   m  m  g   g   y  y                 (b)lue, (m)agenta, (g)reen, (y)ellow
    cuboid_corners = np.array([[x, -x, -x, x, x, -x, -x, x],
                               [-y, -y, y, y, -y, -y, y, y],
                               [z, z, z, z, -z, -z, -z, -z]])
    cuboid_points_model = PointCloud(cuboid_corners, 'target_model')
    cuboid_points_camera = object_pose.apply(cuboid_points_model)
    result_json['cuboid'] = np.transpose(cuboid_points_camera.data).tolist()

    # projected_cuboid_centroid
    cuboid_centroid_image_coords = project_subpixel(camera_intrinsics, Point(cuboid_centroid, 'camera'))
    result_json['projected_cuboid_centroid'] = cuboid_centroid_image_coords.tolist()

    # projected_cuboid
    cuboid_image_coords = project_subpixel(camera_intrinsics, cuboid_points_camera)
    result_json['projected_cuboid'] = np.transpose(cuboid_image_coords).tolist()

    return result_json


def project_subpixel(camera_intrinsics, point_cloud):
    # modified from CameraIntrinsics.project()
    if not isinstance(point_cloud, PointCloud) and not (isinstance(point_cloud, Point) and point_cloud.dim == 3):
        raise ValueError('Must provide PointCloud or 3D Point object for projection')
    if point_cloud.frame != camera_intrinsics.frame:
        raise ValueError('Cannot project points in frame %s into camera with frame %s' % (
            point_cloud.frame, camera_intrinsics.frame))

    points_proj = camera_intrinsics.proj_matrix.dot(point_cloud.data)
    if len(points_proj.shape) == 1:
        points_proj = points_proj[:, np.newaxis]
    point_depths = np.tile(points_proj[2, :], [3, 1])
    points_proj = np.divide(points_proj, point_depths)

    return points_proj[:2, :].squeeze()


def get_cuboid2d_visibility(cuboid2d, img_width, img_height):
    cuboid_poly = MultiPoint(cuboid2d).convex_hull
    img_poly = MultiPoint([(0, 0), (0, img_height), (img_width, img_height), (img_width, 0)]).convex_hull
    return cuboid_poly.intersection(img_poly).area / cuboid_poly.area


def object_pose_from_json(scene_object_json):
    # type: (dict) -> (np.array, np.array)
    """
    Parses object pose from "location" and "quaternion_xyzw".

    :param scene_object_json: JSON fragment of a single scene object
    :return (translation, rotation) in meters
    """
    quaternion_xyzw = np.array(scene_object_json['quaternion_xyzw'])
    quaternion_wxyz = np.roll(quaternion_xyzw, 1)
    rotation = RigidTransform.rotation_from_quaternion(quaternion_wxyz)
    translation = np.array(scene_object_json['location'])
    return translation, rotation


def pose_transform_permuted_to_json(translation, rotation):
    # type: (np.array, np.array) -> dict
    """
    :param translation: location (in meters) as 3x1 array
    :param rotation: rotation as 3x3 matrix
    :return JSON dictionary of pose_transform_permuted (in cm)
    """
    pose_transform_permuted = np.eye(4)
    pose_transform_permuted[:3, 3] = translation
    to_lefthand = np.array([[0, 1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    pose_transform_permuted[:3, :3] = rotation.dot(to_lefthand)
    pose_transform_permuted = np.transpose(pose_transform_permuted)
    return {u'pose_transform': pose_transform_permuted.tolist()}


if __name__ == "__main__":
    main()
