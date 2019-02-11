#!/usr/bin/env python

import numpy as np

try:
   import cPickle as pickle
except:
   import pickle

with open('visibilities.pickle', 'r') as picklefile:
    visibilities = pickle.load(picklefile)

dope_classes = [
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '009_gelatin_box',
    '010_potted_meat_can'
    ]

all_cams = {}

for (object_class, cameras) in visibilities.iteritems():
    if object_class not in dope_classes:
        continue
    for (camera, vis_list) in cameras.iteritems():
        if not all_cams.has_key(camera):
            all_cams[camera] = []
        all_cams[camera] += vis_list

for (camera, vis_list) in all_cams.iteritems():
    print '  {:15} {:.3f}  std {:.3f} {}'.format(camera, np.average(vis_list), np.std(vis_list), len(vis_list))
