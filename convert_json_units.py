#!/usr/bin/env python
import argparse
from copy import deepcopy as dc
import json
import numpy as np

parser = argparse.ArgumentParser(description='Convert translation units in NVidia Dataset format.')
parser.add_argument('infile', type=argparse.FileType('r'),
                    help='json file to convert (frame annotation or object settings)')
parser.add_argument('outfile', type=argparse.FileType('w'), help='output filename')
parser.add_argument('--object-settings-mode', action='store_true',
                    help='process a _object_settings.json file. If false (default), process a frame annotations file.')
parser.add_argument('--unit-scaling', type=float, default=100.0,
                    help='scaling factor for depth units (e.g. 100 to convert from m to cm)')
args = parser.parse_args()


def scale_vector(vector):
    return (np.array(vector) * args.unit_scaling).tolist()


def scale_matrix(matrix):
    output = np.array(matrix)
    output[3, :3] *= args.unit_scaling
    return output.tolist()


def scale_transformation_matrix(matrix):
    output = np.array(matrix) * args.unit_scaling
    output[3, 3] = 1
    return output.tolist()


if args.object_settings_mode:
    # TODO
    in_json = json.load(args.infile)

    # copy relevant fields of json
    out_json = {
        'exported_object_classes': dc(in_json['exported_object_classes']),
        'exported_objects': []
    }

    for i in in_json['exported_objects']:
        out_json['exported_objects'].append({
            'class': dc(i['class']),
            'segmentation_class_id': dc(i['segmentation_class_id']),
            'fixed_model_transform': scale_transformation_matrix(i['fixed_model_transform']),
            'cuboid_dimensions': scale_vector(i['cuboid_dimensions'])
        })

    json.dump(out_json, args.outfile, indent=2, sort_keys=True)

else:
    in_json = json.load(args.infile)

    # copy relevant fields of json
    out_json = {'camera_data': {}, 'objects': []}
    if in_json['camera_data']:
        out_json['camera_data']['location_worldframe'] = scale_vector(
            in_json['camera_data']['location_worldframe'])
        out_json['camera_data']['quaternion_xyzw_worldframe'] = dc(
            in_json['camera_data']['quaternion_xyzw_worldframe'])
    else:
        print 'Warning: no `camera_data` field!'


    for i in in_json['objects']:
        out_json['objects'].append({
            'class': dc(i['class']),
            'visibility': dc(i['visibility']),
            'location': scale_vector(i['location']),
            'quaternion_xyzw': dc(i['quaternion_xyzw']),
            'pose_transform_permuted': scale_matrix(i['pose_transform_permuted']),
            'cuboid_centroid': scale_vector(i['cuboid_centroid']),
            'projected_cuboid_centroid': dc(i['projected_cuboid_centroid']),
            'bounding_box': dc(i['bounding_box']),
            'cuboid': scale_vector(i['cuboid']),
            'projected_cuboid': dc(i['projected_cuboid'])
        })

    json.dump(out_json, args.outfile, indent=2, sort_keys=True)
