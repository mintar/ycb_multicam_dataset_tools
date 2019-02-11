#!/usr/bin/env python

import json
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle

visibilities = {}

with open('files.txt', 'r') as files:
    while True:
        filename = files.readline().strip()
        print filename
        if not filename:
            break
        camera = filename.split('/')[1]
        with open(filename, 'r') as json_file:
            frame_json = json.load(json_file)
        for object_json in frame_json['objects']:
            object_class = object_json['class']
            visibility = object_json['visibility']
            if not visibilities.has_key(object_class):
                visibilities[object_class] = {}
            if not visibilities[object_class].has_key(camera):
                visibilities[object_class][camera] = []
            visibilities[object_class][camera].append(visibility)

for (object_class, cameras) in visibilities.iteritems():
    print '=============', object_class
    for (camera, vis_list) in cameras.iteritems():
        print '  {:15} {:.3f}'.format(camera, np.average(vis_list))
    print ''

with open('visibilities.pickle', 'w') as picklefile:
    pickle.dump(visibilities, picklefile)
