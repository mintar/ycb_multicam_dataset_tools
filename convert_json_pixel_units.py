#!/usr/bin/env python
import argparse
from copy import deepcopy as dc
import json
import numpy as np

parser = argparse.ArgumentParser(description='Convert pixel-based units in NVidia Dataset format.')
parser.add_argument('infile', type=argparse.FileType('r'),
                    help='json file to convert (frame annotation or camera settings)')
parser.add_argument('outfile', type=argparse.FileType('w'), help='output filename')
parser.add_argument('--camera-settings-mode', action='store_true',
                    help='process a _camera_settings.json file. If false (default), process a frame annotations file.')
parser.add_argument('--pixel-scaling', type=float, default=(400.0 / 1024.0),
                    help='scaling factor from intput to output image (e.g. 0.390625 to convert from 1024px to 400 px)')
args = parser.parse_args()


def scale_pixel_vector(vector):
    '''
    Scales a vector (or matrix) that is in pixel coordinates
    '''
    return (np.array(vector) * args.pixel_scaling).tolist()

in_json = json.load(args.infile)

if args.camera_settings_mode:
    # copy relevant fields of json
    out_json = {'camera_settings': []}
    for i in in_json['camera_settings']:
        out_cam_json = {
            'name': dc(i['name']),
            'horizontal_fov': dc(i['horizontal_fov']),
            'intrinsic_settings': {
                'resX': int(i['intrinsic_settings']['resX'] * args.pixel_scaling),
                'resY': int(i['intrinsic_settings']['resY'] * args.pixel_scaling),
                'fx': i['intrinsic_settings']['fx'] * args.pixel_scaling,
                'fy': i['intrinsic_settings']['fy'] * args.pixel_scaling,
                'cx': i['intrinsic_settings']['cx'] * args.pixel_scaling,
                'cy': i['intrinsic_settings']['cy'] * args.pixel_scaling,
                's': i['intrinsic_settings']['s'] * args.pixel_scaling
            },
            'captured_image_size': {
                'width': int(i['captured_image_size']['width'] * args.pixel_scaling),
                'height': int(i['captured_image_size']['height'] * args.pixel_scaling)
            }
        }
        out_json['camera_settings'].append(out_cam_json)
    
else:

    # copy relevant fields of json
    out_json = {'camera_data': {}, 'objects': []}
    if in_json['camera_data']:
        out_json['camera_data'] = dc(in_json['camera_data'])
    else:
        print 'Warning: no `camera_data` field!'


    for i in in_json['objects']:
        out_object_json = {
            'class': dc(i['class']),
            'visibility': dc(i['visibility']),
            'location': dc(i['location']),
            'quaternion_xyzw': dc(i['quaternion_xyzw']),
            'cuboid_centroid': dc(i['cuboid_centroid']),
            'projected_cuboid_centroid': scale_pixel_vector(i['projected_cuboid_centroid']),
            'bounding_box': {'bottom_right': scale_pixel_vector(i['bounding_box']['bottom_right']),
                             'top_left': scale_pixel_vector(i['bounding_box']['top_left'])},
            'cuboid': dc(i['cuboid']),
            'projected_cuboid': scale_pixel_vector(i['projected_cuboid'])
        }

        # optional fields
        try:
            out_object_json['instance_id'] = dc(i['instance_id'])
        except KeyError:
            pass

        try:
            out_object_json['pose_transform_permuted'] = dc(i['pose_transform_permuted'])
        except KeyError:
            pass

        out_json['objects'].append(out_object_json)


json.dump(out_json, args.outfile, indent=2, sort_keys=True)
