#!/usr/bin/env python
import json
import argparse
import os
import sys
import re
import imageio
import copy
import numpy as np
import trimesh
import collections
from shapely.geometry import MultiPoint
from autolab_core import RigidTransform, PointCloud, Point
from perception import CameraIntrinsics, RenderMode

# os.environ['MESHRENDER_EGL_OFFSCREEN'] = 't'
from meshrender import Scene, MaterialProperties, SceneObject, VirtualCamera, SceneViewer

ModelConfig = collections.namedtuple('ModelConfig',
                                     'class_name segmentation_class_id model_transform cuboid_dimensions mesh')

MAX_DEPTH_DIFF = 0.04  # difference allowed between synthetic and real depth image


def main():
    # ====================================
    # parse arguments
    # ====================================
    parser = argparse.ArgumentParser(description='Generate segmentation images and updated json files.')
    parser.add_argument('-d', '--data-dir', default=os.getcwd(), help='directory containing images and json files')
    parser.add_argument('-m', '--mesh-dir', required=True, help='directory containing the mesh files')
    parser.add_argument('-o', '--object-settings',
                        help='object_settings file where the fixed_model_transform corresponds to the poses'
                             'in the frame annotation jsons.'
                             'default: <data_dir>/_object_settings.json')
    parser.add_argument('-t', '--target-object-settings',
                        help='if given, transform all poses into the fixed_model_transform from this file.')
    parser.add_argument('--unit-scaling', type=float, default=1.0,
                        help='scaling factor for depth units (e.g. 100.0 to convert from m to cm)')
    parser.add_argument('--mesh-scaling', type=float, default=0.01,
                        help='scaling factor for meshes (e.g. 100.0 to convert from m to cm)')
    parser.add_argument('--no-save-vertmap', action='store_true',
                        help='Do not save vertmap.npz files. vertmap files are useful for PoseCNN training.')
    parser.add_argument('--gui', action='store_true', help='Start a GUI after rendering each depth image.')
    args = parser.parse_args()

    if not args.object_settings:
        args.object_settings = os.path.join(args.data_dir, '_object_settings.json')
    if not args.target_object_settings:
        args.target_object_settings = args.object_settings

    # =====================================
    # parse object_settings and load meshes
    # =====================================
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
        if class_name.endswith('_16k'):
            mesh_path = os.path.join(args.mesh_dir, class_name[:-4], 'google_16k/textured.obj')
        else:
            mesh_path = os.path.join(args.mesh_dir, class_name, 'google_16k/textured.obj')
        segmentation_class_id = model_json['segmentation_class_id']
        cuboid_dimensions = np.array(model_json['cuboid_dimensions'])
        mesh = trimesh.load_mesh(mesh_path)
        mesh.apply_scale(args.mesh_scaling)

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

        models[class_name] = ModelConfig(class_name, segmentation_class_id, model_transform, cuboid_dimensions,
                                         mesh)

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
    pattern = re.compile(r'\d{3,}.ycbm.json')
    json_files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if pattern.match(f)])
    for json_file in json_files:
        filename_prefix = json_file[:-len('ycbm.json')]
        print '\n---------------------- {}*'.format(filename_prefix)
        with open(json_file, 'r') as f:
            frame_json = json.load(f)

        real_depth_image = np.expand_dims(imageio.imread(filename_prefix + 'depth.png'), 2) / (
                10000.0 / args.unit_scaling)
        segmentation_image, updated_frame_json, vertmap = process_frame(frame_json,
                                                                        real_depth_image,
                                                                        camera_intrinsics,
                                                                        models,
                                                                        args.unit_scaling,
                                                                        args.gui)
        if segmentation_image is not None:
            imageio.imwrite(filename_prefix + 'seg.png', segmentation_image)
        with open(filename_prefix + 'ycbm_full.json', 'w') as f:
            json.dump(updated_frame_json, f, indent=2, sort_keys=True)
        if not args.no_save_vertmap:
            np.savez_compressed(filename_prefix + 'vertmap.npz', vertmap=vertmap)


def process_frame(frame_json, real_depth_image, camera_intrinsics, models, unit_scaling, start_viewer=False):
    num_objects = len(frame_json['objects'])
    if len(frame_json['objects']) == 0:
        print "no objects in frame!"
        return None, None

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

    # get object poses and render separate depth images
    object_poses = []
    depth_images = []
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
        object_poses.append(object_pose)

        updated_frame_json['objects'][object_index]['location'] = object_pose.translation.tolist()
        updated_frame_json['objects'][object_index]['quaternion_xyzw'] = np.roll(object_pose.quaternion, -1).tolist()

        mesh = models[scene_object_json['class']].mesh
        depth_images.append(render_depth_image(object_pose, camera_intrinsics, mesh, unit_scaling, start_viewer))

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

    # overlay depth images, calculate segmentation image, visibility, bounding_box
    segmentation_image, updated_frame_json, vertmap = get_segmentation_image(updated_frame_json,
                                                                             depth_images,
                                                                             real_depth_image,
                                                                             camera_intrinsics,
                                                                             object_poses,
                                                                             models)

    # remove objects with zero visibility
    for obj in list(updated_frame_json['objects']):  # temporary copy for deletion while iterating
        if obj['visibility'] == 0.0:
            print "Removing object because of zero visibility: {}".format(obj['class'])
            updated_frame_json['objects'].remove(obj)

    return segmentation_image, updated_frame_json, vertmap


def get_segmentation_image(frame_json, depth_images, real_depth_image, camera_intrinsics, object_poses, models):
    updated_frame_json = copy.deepcopy(frame_json)
    num_objects = len(frame_json['objects'])

    # calculate bounding_box and update json
    for object_index in range(num_objects):
        depth_image = depth_images[object_index].raw_data
        cols = np.any(depth_image, axis=0)
        rows = np.any(depth_image, axis=1)
        if not np.any(cols):
            # depth image is empty, so bounding box is undefined
            continue
        umin, umax = np.nonzero(cols)[0][[0, -1]]
        vmin, vmax = np.nonzero(rows)[0][[0, -1]]

        # TODO: remove this quirk of the FAT dataset; corrds should be (u, v), not (v, u)
        bbox_top_left = [vmin, umin]
        bbox_bottom_right = [vmax, umax]

        updated_frame_json['objects'][object_index]['bounding_box'] = {}
        updated_frame_json['objects'][object_index]['bounding_box']['top_left'] = bbox_top_left
        updated_frame_json['objects'][object_index]['bounding_box']['bottom_right'] = bbox_bottom_right

    # calculate object index image
    object_index_image = np.full_like(depth_images[0].raw_data, 255, dtype=np.uint8)  # 255 = invalid object index
    combined_depth_image = np.full_like(depth_images[0].raw_data, np.inf)
    for object_index in range(num_objects):
        depth_image = depth_images[object_index].raw_data
        mask = np.logical_and(depth_image < combined_depth_image, depth_image != 0)
        combined_depth_image = np.choose(mask, (combined_depth_image, depth_image))
        object_index_image = np.choose(mask, (object_index_image, object_index))

    visible_pixels_before_filtering = num_objects * [0]
    for object_index in range(num_objects):
        visible_pixels_before_filtering[object_index] = int(np.sum(object_index_image == object_index))

    # filter segmentation image by comparison with real_depth_image
    within_depth_diff_mask = np.logical_or(real_depth_image == 0.0,
                                           np.absolute(combined_depth_image - real_depth_image) < MAX_DEPTH_DIFF)
    object_index_image = np.choose(within_depth_diff_mask, (255, object_index_image))

    # calculate visibility and update json
    for object_index in range(num_objects):
        total_pixels = np.sum(depth_images[object_index].raw_data > 0)
        visible_pixels = np.sum(object_index_image == object_index)

        if total_pixels == 0:
            visibility = 0.0
            ground_truth_mismatch = 0.0
        else:
            visibility = float(visible_pixels) / total_pixels
            ground_truth_mismatch = float(visible_pixels_before_filtering[object_index] - visible_pixels) / total_pixels

        # adjust visibility based on fraction of cuboid in camera frustum
        visibility *= get_cuboid2d_visibility(updated_frame_json['objects'][object_index]['projected_cuboid'],
                                              depth_images[object_index].width, depth_images[object_index].height)

        updated_frame_json['objects'][object_index]['visibility'] = visibility
        updated_frame_json['objects'][object_index]['ground_truth_mismatch'] = ground_truth_mismatch
        if ground_truth_mismatch > 0.1:
            print "{:26} visibility: {:1.3f}   ground_truth_mismatch: {:1.3f}".format(
                updated_frame_json['objects'][object_index]['class'], visibility, ground_truth_mismatch)
        else:
            print "{:26} visibility: {:1.3f}".format(
                updated_frame_json['objects'][object_index]['class'], visibility)

    # convert object_index_image to segmentation_image
    segmentation_image = np.zeros_like(object_index_image)
    for object_index in range(num_objects):
        segmentation_id = models[frame_json['objects'][object_index]['class']].segmentation_class_id
        segmentation_image = np.where(object_index_image == object_index, segmentation_id, segmentation_image)

    # compute vertmap
    vertmap = np.zeros((object_index_image.shape[0], object_index_image.shape[1], 3), dtype=np.float32)
    for object_index in range(num_objects):
        points_camera = camera_intrinsics.deproject(depth_images[object_index])
        points_model = object_poses[object_index].inverse().apply(points_camera)
        points_model_data = np.transpose(points_model.data).reshape(vertmap.shape)
        vertmap = np.where(object_index_image == object_index, points_model_data, vertmap)

    return segmentation_image, updated_frame_json, vertmap


# def get_segmentation_image_slow(frame_json, depth_images, segmentation_class_ids):
#     updated_frame_json = copy.deepcopy(frame_json)
#     num_objects = len(frame_json['objects'])
#
#     # reverse segmentation_class_ids
#     segmentation_id_to_object_index = {}
#     for object_index in range(num_objects):
#         for object_class, segmentation_id in segmentation_class_ids.iteritems():
#             if object_class == frame_json['objects'][object_index]['class']:
#                 segmentation_id_to_object_index[segmentation_id] = object_index
#                 # this assumes that there are no two objects with the same class
#
#     # calculate segmentation image
#     segmentation_image = np.zeros_like(depth_images[0].raw_data, dtype=np.uint8)
#     combined_depth_image = np.zeros_like(depth_images[0].raw_data)
#     total_pixels = num_objects * [0]
#     hidden_pixels = num_objects * [0]
#     for object_index in range(num_objects):
#         for i, depth in np.ndenumerate(depth_images[object_index].raw_data):
#             if depth > 0.0:
#                 total_pixels[object_index] += 1
#                 if combined_depth_image[i] == 0:
#                     combined_depth_image[i] = depth
#                     segmentation_image[i] = segmentation_class_ids[
#                         frame_json['objects'][object_index]['class']]
#                 else:
#                     if depth > combined_depth_image[i]:
#                         # current object is occluded by an object already in depth image
#                         hidden_pixels[object_index] += 1
#                     else:
#                         # current object is occluding an object already in depth image
#                         occluded_object = segmentation_id_to_object_index[segmentation_image[i]]
#                         hidden_pixels[occluded_object] += 1
#                         combined_depth_image[i] = depth
#                         segmentation_image[i] = segmentation_class_ids[
#                             frame_json['objects'][object_index]['class']]
#
#     # update visibility in json
#     for object_index in range(num_objects):
#         if total_pixels[object_index] == 0:
#             visibility = 0.0
#         else:
#             visibility = float(total_pixels[object_index] - hidden_pixels[object_index]) / total_pixels[object_index]
#         frame_json['objects'][object_index]['visibility'] = visibility
#         print "visibility: ", visibility
#
#     print '\n----------------------\n'
#
#     return segmentation_image, updated_frame_json


def render_depth_image(object_pose, camera_intrinsics, mesh, unit_scaling, start_viewer):
    scene = Scene()

    # Set up a material property (only used for visualization)
    blue_material = MaterialProperties(
        color=np.array([0.1, 0.1, 0.5]),
        k_a=0.3,
        k_d=1.0,
        k_s=1.0,
        alpha=10.0,
        smooth=False
    )
    scene_obj = SceneObject(mesh=mesh, T_obj_world=object_pose, material=blue_material)
    scene.add_object('target_model', scene_obj)

    camera_pose = RigidTransform(from_frame='camera', to_frame='world')
    scene.camera = VirtualCamera(camera_intrinsics, camera_pose, z_near=(0.05 * unit_scaling),
                                 z_far=(6.5535 * unit_scaling))
    if start_viewer:
        SceneViewer(scene, raymond_lighting=True, starting_camera_pose=camera_pose)

    [depth_image] = scene.wrapped_render([RenderMode.DEPTH])
    # imageio.imwrite('depth.png', (depth_image.raw_data * DEPTH_SCALING_FACTOR).astype(np.uint16))

    return depth_image


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


def pose_transform_permuted_from_json(scene_object_json):
    # type: (dict) -> (np.array, np.array)
    """
    Parses object pose from "pose_transform_permuted". Equivalent to parse_object_pose.

    *Note:*  Like the `fixed_model_transform`, the `pose_transform_permuted` is actually the transpose of the matrix.
    Moreover, after transposing, the columns are permuted, and there is a sign flip (due to UE4's use of a lefthand
    coordinate system).  Specifically, if `A` is the matrix given by `pose_transform_permuted`, then actual transform
    is given by `A^T * P`, where `^T` denotes transpose, `*` denotes matrix multiplication, and the permutation matrix
    `P` is given by

        [ 0  0  1]
    P = [ 1  0  0]
        [ 0 -1  0]

    :param scene_object_json: JSON fragment of a single scene object
    :return (translation, rotation) in meters
    """
    pose_transform = np.transpose(np.array(scene_object_json['pose_transform_permuted']))
    to_righthand = np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, -1, 0]])
    rotation = pose_transform[:3, :3].dot(to_righthand)
    translation = pose_transform[:3, 3]
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
    return {u'pose_transform_permuted': pose_transform_permuted.tolist()}


if __name__ == "__main__":
    main()
