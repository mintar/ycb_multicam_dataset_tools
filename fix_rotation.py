#!/usr/bin/env python
import json
import argparse
import sys
import numpy as np
import copy
from autolab_core import RigidTransform

"""
Usage:

    # make backup of complete folder!

    for f in *.ycbm.json; do fix_rotation.py $f $(basename $f .ycbm.json).ycbm_fixed.json; done
    for f in *.ycbm.json; do mv $(basename $f .ycbm.json).ycbm_fixed.json $f; done
    process_single_folder.sh
"""


##### Fix ground truth of 005_011_021_025_035_036_040_061
## json of the incorrect ground truth (from .rgb.json)
#reference_source_json = json.loads("""
#{
#  "objects": [
#    {
#      "class": "005_tomato_soup_can",
#      "location": [7.418785388582336, -1.0361944875518767, 64.74901490141752],
#      "quaternion_xyzw": [0.16270705752384268, 0.6023353302663955, 0.4022562500281078, 0.6700062422296575]
#    }
#  ]
#}
#""")
#
## json of the correct object detection (from .dope_eval.json)
#reference_target_json = json.loads("""
#{
#  "objects": [
#    {
#      "class": "005_tomato_soup_can",
#      "location": [7.417383505276895, -0.6876149830104283, 64.98855260490068],
#      "quaternion_xyzw": [0.3682408029635768, -0.3030776589187064, 0.06792181998966908, 0.8763157365164609]
#    }
#  ]
#}
#""")

#### Fix ground truth of 005_006_009_010_036 (well visible in astra cam frame 40)
# json of the incorrect ground truth (from .rgb.json)
reference_source_json = json.loads("""
{
  "objects": [
    {
      "class": "010_potted_meat_can",
      "location": [28.38215077837108, 11.502942511235831, 63.60884030827886],
      "quaternion_xyzw": [0.1000541438013984, 0.9371587492626866, 0.2858480163707755, 0.17324353499921902]
    }
  ]
}
""")

# json of the correct object detection (from .dope_eval.json)
reference_target_json = json.loads("""
{
  "objects": [
    {
      "class": "010_potted_meat_can",
      "location": [28.172191381792604, 11.405770070550957, 62.36138450176089],
      "quaternion_xyzw": [0.2929615724554571, -0.16685066733120527, -0.0780634575881229, 0.9382113133324202]
    }
  ]
}
""")

############## NO NEED TO CHANGE BELOW THIS LINE ####################


# ycb_object_settings.json
model_json = json.loads("""
{
  "exported_object_classes": [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick"
  ],
  "exported_objects": [
    {
      "class": "002_master_chef_can",
      "segmentation_class_id": 12,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.10240400314331055, 0.14017700195312499, 0.10230899810791015 ]
    },
    {
      "class": "003_cracker_box",
      "segmentation_class_id": 24,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.16403600692749024, 0.21343700408935548, 0.07179999828338623 ]
    },
    {
      "class": "004_sugar_box",
      "segmentation_class_id": 36,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.09267700195312500, 0.17625299453735352, 0.04513400077819824 ]
    },
    {
      "class": "005_tomato_soup_can",
      "segmentation_class_id": 48,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.06765900135040283, 0.10185500144958497, 0.06771399974822997 ]
    },
    {
      "class": "006_mustard_bottle",
      "segmentation_class_id": 60,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.09602399826049805, 0.19130100250244142, 0.05824900150299072 ]
    },
    {
      "class": "007_tuna_fish_can",
      "segmentation_class_id": 72,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.08555899620056152, 0.03353800058364868, 0.08553999900817871 ]
    },
    {
      "class": "008_pudding_box",
      "segmentation_class_id": 84,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.11365400314331055, 0.08981699943542480, 0.03847100019454956 ]
    },
    {
      "class": "009_gelatin_box",
      "segmentation_class_id": 96,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.08918299674987792, 0.07311500072479248, 0.02998300075531006 ]
    },
    {
      "class": "010_potted_meat_can",
      "segmentation_class_id": 108,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.10164699554443360, 0.08354299545288085, 0.05760099887847901 ]
    },
    {
      "class": "011_banana",
      "segmentation_class_id": 120,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.19717399597167970, 0.03864900112152100, 0.07406599998474121 ]
    },
    {
      "class": "019_pitcher_base",
      "segmentation_class_id": 132,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.18861799240112304, 0.24238700866699220, 0.13310600280761720 ]
    },
    {
      "class": "021_bleach_cleanser",
      "segmentation_class_id": 144,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.10243300437927245, 0.25058000564575195, 0.06769899845123291 ]
    },
    {
      "class": "024_bowl",
      "segmentation_class_id": 156,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.16116399765014650, 0.05500899791717529, 0.16146299362182617 ]
    },
    {
      "class": "025_mug",
      "segmentation_class_id": 168,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.11699000358581543, 0.08130299568176269, 0.09308799743652343 ]
    },
    {
      "class": "035_power_drill",
      "segmentation_class_id": 180,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.18436100006103515, 0.18683599472045898, 0.05719200134277344 ]
    },
    {
      "class": "036_wood_block",
      "segmentation_class_id": 192,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.08969599723815919, 0.20609199523925781, 0.09078000068664550 ]
    },
    {
      "class": "037_scissors",
      "segmentation_class_id": 204,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.20259799957275390, 0.08753000259399414, 0.01565500020980835 ]
    },
    {
      "class": "040_large_marker",
      "segmentation_class_id": 216,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.01883399963378906, 0.12088600158691407, 0.01947600007057190 ]
    },
    {
      "class": "051_large_clamp",
      "segmentation_class_id": 228,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.16494100570678710, 0.12166600227355957, 0.03641299962997437 ]
    },
    {
      "class": "052_extra_large_clamp",
      "segmentation_class_id": 240,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.20259599685668944, 0.16520299911499023, 0.03647599935531616 ]
    },
    {
      "class": "061_foam_brick",
      "segmentation_class_id": 252,
      "fixed_model_transform": [
        [ 1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
      ],
      "cuboid_dimensions": [ 0.07787399768829345, 0.05119299888610840, 0.05256000041961670 ]
    }
  ]
}
""")

# aligned_m_object_settings.json
target_model_json = json.loads("""
{
  "exported_object_classes": [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick"
  ],
  "exported_objects": [
    {
      "class": "002_master_chef_can",
      "cuboid_dimensions": [
        0.10240400314331055,
        0.140177001953125,
        0.10230899810791017
      ],
      "fixed_model_transform": [
        [
          0.8660250091552735,
          0.0,
          0.5,
          0.0
        ],
        [
          -0.5,
          0.0,
          0.8660250091552735,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.009908000230789185,
          0.06990200042724609,
          0.01690299987792969,
          1.0
        ]
      ],
      "segmentation_class_id": 12
    },
    {
      "class": "003_cracker_box",
      "cuboid_dimensions": [
        0.16403600692749024,
        0.21343700408935548,
        0.07179999828338623
      ],
      "fixed_model_transform": [
        [
          0.0,
          0.0,
          1.0,
          0.0
        ],
        [
          -1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          -0.01414199948310852,
          0.1034749984741211,
          0.0128849995136261,
          1.0
        ]
      ],
      "segmentation_class_id": 24
    },
    {
      "class": "004_sugar_box",
      "cuboid_dimensions": [
        0.092677001953125,
        0.17625299453735352,
        0.04513400077819824
      ],
      "fixed_model_transform": [
        [
          -0.03487799882888794,
          0.034899001121520994,
          0.998781967163086,
          0.0
        ],
        [
          -0.9992600250244141,
          -0.017441999912261964,
          -0.03428499937057495,
          0.0
        ],
        [
          0.016224000453948974,
          -0.9992389678955078,
          0.035481998920440676,
          0.0
        ],
        [
          -0.01795199990272522,
          0.0875790023803711,
          0.0038839998841285707,
          1.0
        ]
      ],
      "segmentation_class_id": 36
    },
    {
      "class": "005_tomato_soup_can",
      "cuboid_dimensions": [
        0.06765900135040283,
        0.10185500144958497,
        0.06771399974822999
      ],
      "fixed_model_transform": [
        [
          0.9914450073242188,
          0.0,
          -0.13052599906921386,
          0.0
        ],
        [
          0.13052599906921386,
          0.0,
          0.9914450073242188,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          -0.001793999969959259,
          0.05100699901580811,
          -0.08444399833679199,
          1.0
        ]
      ],
      "segmentation_class_id": 48
    },
    {
      "class": "006_mustard_bottle",
      "cuboid_dimensions": [
        0.09602399826049805,
        0.19130100250244142,
        0.05824900150299073
      ],
      "fixed_model_transform": [
        [
          0.9205049896240235,
          0.0,
          0.39073101043701175,
          0.0
        ],
        [
          -0.39073101043701175,
          0.0,
          0.9205049896240235,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.004925999939441681,
          0.09249799728393555,
          0.027135999202728273,
          1.0
        ]
      ],
      "segmentation_class_id": 60
    },
    {
      "class": "007_tuna_fish_can",
      "cuboid_dimensions": [
        0.08555899620056152,
        0.03353800058364868,
        0.08553999900817871
      ],
      "fixed_model_transform": [
        [
          1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          1.0,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.026048998832702636,
          0.013551000356674194,
          0.022132000923156737,
          1.0
        ]
      ],
      "segmentation_class_id": 72
    },
    {
      "class": "008_pudding_box",
      "cuboid_dimensions": [
        0.11365400314331055,
        0.0898169994354248,
        0.03847100019454956
      ],
      "fixed_model_transform": [
        [
          -0.8826450347900391,
          0.46947200775146486,
          -0.023113000392913818,
          0.0
        ],
        [
          -0.4687820053100586,
          -0.8828130340576172,
          -0.02973400115966797,
          0.0
        ],
        [
          -0.03436300039291382,
          -0.015410000085830688,
          0.999291000366211,
          0.0
        ],
        [
          0.01010200023651123,
          0.016993999481201172,
          -0.01757200002670288,
          1.0
        ]
      ],
      "segmentation_class_id": 84
    },
    {
      "class": "009_gelatin_box",
      "cuboid_dimensions": [
        0.08918299674987794,
        0.07311500072479248,
        0.02998300075531006
      ],
      "fixed_model_transform": [
        [
          0.22494199752807617,
          0.9743699645996095,
          0.001962999999523163,
          0.0
        ],
        [
          -0.9743329620361328,
          0.22495100021362305,
          -0.008503000140190125,
          0.0
        ],
        [
          -0.008726999759674073,
          0.0,
          0.9999620056152344,
          0.0
        ],
        [
          -0.002906999886035919,
          0.023998000621795655,
          -0.014543999433517456,
          1.0
        ]
      ],
      "segmentation_class_id": 96
    },
    {
      "class": "010_potted_meat_can",
      "cuboid_dimensions": [
        0.1016469955444336,
        0.08354299545288087,
        0.057600998878479005
      ],
      "fixed_model_transform": [
        [
          0.9986299896240235,
          0.0,
          -0.0523360013961792,
          0.0
        ],
        [
          0.0523360013961792,
          0.0,
          0.9986299896240235,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.03406599998474121,
          0.03858400106430054,
          0.024767000675201416,
          1.0
        ]
      ],
      "segmentation_class_id": 108
    },
    {
      "class": "011_banana",
      "cuboid_dimensions": [
        0.1971739959716797,
        0.038649001121521,
        0.07406599998474121
      ],
      "fixed_model_transform": [
        [
          -0.3639540100097656,
          -0.17364799499511718,
          0.9150869750976562,
          0.0
        ],
        [
          -0.9308769989013672,
          0.03436899900436401,
          -0.36371200561523437,
          0.0
        ],
        [
          0.03170700073242187,
          -0.9842079925537109,
          -0.17415399551391603,
          0.0
        ],
        [
          0.0003660000115633011,
          0.014974999427795411,
          0.004444999992847443,
          1.0
        ]
      ],
      "segmentation_class_id": 120
    },
    {
      "class": "019_pitcher_base",
      "cuboid_dimensions": [
        0.18861799240112306,
        0.2423870086669922,
        0.1331060028076172
      ],
      "fixed_model_transform": [
        [
          0.7193399810791016,
          0.0,
          -0.6946579742431641,
          0.0
        ],
        [
          0.6946579742431641,
          0.0,
          0.7193399810791016,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          -0.02433000087738037,
          0.118371000289917,
          -0.03410599946975708,
          1.0
        ]
      ],
      "segmentation_class_id": 132
    },
    {
      "class": "021_bleach_cleanser",
      "cuboid_dimensions": [
        0.10243300437927247,
        0.25058000564575195,
        0.06769899845123291
      ],
      "fixed_model_transform": [
        [
          -1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          0.0,
          -1.0,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          -0.02166399955749512,
          0.12483799934387207,
          0.011708999872207642,
          1.0
        ]
      ],
      "segmentation_class_id": 144
    },
    {
      "class": "024_bowl",
      "cuboid_dimensions": [
        0.1611639976501465,
        0.05500899791717529,
        0.16146299362182617
      ],
      "fixed_model_transform": [
        [
          0.0,
          0.0,
          1.0,
          0.0
        ],
        [
          -1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          -0.043688998222351075,
          0.026974000930786134,
          0.014838999509811402,
          1.0
        ]
      ],
      "segmentation_class_id": 156
    },
    {
      "class": "025_mug",
      "cuboid_dimensions": [
        0.11699000358581543,
        0.08130299568176269,
        0.09308799743652343
      ],
      "fixed_model_transform": [
        [
          0.9986289978027344,
          0.0,
          0.0523360013961792,
          0.0
        ],
        [
          -0.0523360013961792,
          0.0,
          0.9986289978027344,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.009767000079154969,
          0.04011899948120117,
          -0.016074999570846557,
          1.0
        ]
      ],
      "segmentation_class_id": 168
    },
    {
      "class": "035_power_drill",
      "cuboid_dimensions": [
        0.18436100006103515,
        0.18683599472045898,
        0.05719200134277344
      ],
      "fixed_model_transform": [
        [
          0.9992389678955078,
          0.017452000379562377,
          -0.03489399909973145,
          0.0
        ],
        [
          0.01652199983596802,
          -0.9995050048828126,
          -0.026770000457763673,
          0.0
        ],
        [
          -0.0353439998626709,
          0.02617300033569336,
          -0.9990319824218751,
          0.0
        ],
        [
          0.045402998924255374,
          0.011200000047683716,
          0.02299099922180176,
          1.0
        ]
      ],
      "segmentation_class_id": 180
    },
    {
      "class": "036_wood_block",
      "cuboid_dimensions": [
        0.08969599723815919,
        0.2060919952392578,
        0.0907800006866455
      ],
      "fixed_model_transform": [
        [
          -0.22494199752807617,
          0.008726999759674073,
          0.9743329620361328,
          0.0
        ],
        [
          -0.9743699645996095,
          0.0,
          -0.22495100021362305,
          0.0
        ],
        [
          -0.001962999999523163,
          -0.9999620056152344,
          0.008503000140190125,
          0.0
        ],
        [
          -0.005261999964714051,
          0.10234100341796876,
          -0.02628200054168701,
          1.0
        ]
      ],
      "segmentation_class_id": 192
    },
    {
      "class": "037_scissors",
      "cuboid_dimensions": [
        0.2025979995727539,
        0.08753000259399414,
        0.01565500020980835
      ],
      "fixed_model_transform": [
        [
          -0.2752599906921387,
          0.961261978149414,
          -0.014426000118255615,
          0.0
        ],
        [
          -0.9613680267333985,
          -0.2752599906921387,
          0.0020250000059604647,
          0.0
        ],
        [
          -0.0020250000059604647,
          0.014426000118255615,
          0.9998940277099609,
          0.0
        ],
        [
          0.045185999870300295,
          -0.012343000173568725,
          -0.00713100016117096,
          1.0
        ]
      ],
      "segmentation_class_id": 204
    },
    {
      "class": "040_large_marker",
      "cuboid_dimensions": [
        0.018833999633789063,
        0.12088600158691407,
        0.0194760000705719
      ],
      "fixed_model_transform": [
        [
          -0.061027998924255374,
          0.026177000999450684,
          0.99779296875,
          0.0
        ],
        [
          -0.001597999930381775,
          -0.9996569824218751,
          0.02612799882888794,
          0.0
        ],
        [
          0.998134994506836,
          0.0,
          0.061048998832702636,
          0.0
        ],
        [
          -0.011348999738693237,
          -0.004075999855995179,
          0.035237998962402345,
          1.0
        ]
      ],
      "segmentation_class_id": 216
    },
    {
      "class": "051_large_clamp",
      "cuboid_dimensions": [
        0.1649410057067871,
        0.12166600227355957,
        0.036412999629974366
      ],
      "fixed_model_transform": [
        [
          -0.1455399990081787,
          -0.9832550048828125,
          0.1096720027923584,
          0.0
        ],
        [
          0.9875430297851563,
          -0.15107999801635744,
          -0.04398200035095215,
          0.0
        ],
        [
          0.05981400012969971,
          0.10190500259399414,
          0.9929940032958985,
          0.0
        ],
        [
          0.01027400016784668,
          -0.00810100018978119,
          -0.01802299976348877,
          1.0
        ]
      ],
      "segmentation_class_id": 228
    },
    {
      "class": "052_extra_large_clamp",
      "cuboid_dimensions": [
        0.20259599685668947,
        0.16520299911499023,
        0.03647599935531616
      ],
      "fixed_model_transform": [
        [
          0.993572006225586,
          -0.11320300102233886,
          0.0,
          0.0
        ],
        [
          0.11318599700927734,
          0.9934200286865235,
          -0.017452000379562377,
          0.0
        ],
        [
          0.0019760000705718993,
          0.01733999967575073,
          0.9998480224609375,
          0.0
        ],
        [
          0.027757999897003175,
          0.03177799940109253,
          -0.018323999643325806,
          1.0
        ]
      ],
      "segmentation_class_id": 240
    },
    {
      "class": "061_foam_brick",
      "cuboid_dimensions": [
        0.07787399768829346,
        0.0511929988861084,
        0.0525600004196167
      ],
      "fixed_model_transform": [
        [
          0.0,
          0.0,
          1.0,
          0.0
        ],
        [
          -1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          -1.0,
          0.0,
          0.0
        ],
        [
          0.017200000286102295,
          0.02525099992752075,
          0.018260999917984008,
          1.0
        ]
      ],
      "segmentation_class_id": 252
    }
  ]
}
""")


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


parser = argparse.ArgumentParser(description='Fix rotation')
parser.add_argument('infile', type=argparse.FileType('r'), help='json file to convert (frame annotation)')
parser.add_argument('outfile', type=argparse.FileType('w'), help='output filename')
args = parser.parse_args()

MODEL_NAME = reference_source_json['objects'][0]['class']

# calculate model_transform
fmt = {}
for j in model_json['exported_objects']:
    if j['class'] == MODEL_NAME:
        fmt = j['fixed_model_transform']
        break
fixed_model_transform_mat = np.transpose(np.array(fmt))
fixed_model_transform = RigidTransform(
    rotation=fixed_model_transform_mat[:3, :3],
    translation=fixed_model_transform_mat[:3, 3],
    from_frame='ycb_model',
    to_frame='source_model'
)

fmt = {}
for j in target_model_json['exported_objects']:
    if j['class'] == MODEL_NAME:
        fmt = j['fixed_model_transform']
        break
target_fixed_model_transform_mat = np.transpose(np.array(fmt))
target_fixed_model_transform = RigidTransform(
    rotation=target_fixed_model_transform_mat[:3, :3],
    translation=target_fixed_model_transform_mat[:3, 3],
    from_frame='ycb_model',
    to_frame='target_model'
)
model_transform = fixed_model_transform.dot(target_fixed_model_transform.inverse())

(translation, rotation) = object_pose_from_json(reference_source_json['objects'][0])
reference_source_pose = RigidTransform(
    rotation=rotation,
    translation=translation,
    from_frame='target_model',
    to_frame='camera'
)

(translation, rotation) = object_pose_from_json(reference_target_json['objects'][0])
reference_target_pose = RigidTransform(
    rotation=rotation,
    translation=translation,
    from_frame='target_model_fixed',
    to_frame='camera'
)

tgt_model_to_tgt_model_fixed = reference_source_pose.inverse().dot(reference_target_pose)
tgt_model_to_tgt_model_fixed.translation = np.zeros((3,))

model_transform_fixed = copy.deepcopy(model_transform)
model_transform_fixed.from_frame = 'target_model_fixed'
model_transform_fixed.to_frame = 'source_model_fixed'

src_model_to_src_model_fixed = model_transform.dot(tgt_model_to_tgt_model_fixed).dot(model_transform_fixed.inverse())

frame_json = json.load(args.infile)
args.infile.close()

num_objects = len(frame_json['objects'])

if len(frame_json['objects']) == 0:
    print "no objects in frame!"
    sys.exit(1)

for object_index in range(num_objects):
    model_name = frame_json['objects'][object_index]['class']
    if model_name != MODEL_NAME:
        continue

    scene_object_json = frame_json['objects'][object_index]
    (translation, rotation) = object_pose_from_json(scene_object_json)
    object_pose = RigidTransform(
        rotation=rotation,
        translation=translation,
        from_frame='source_model',
        to_frame='camera'
    )
    object_pose = object_pose.dot(src_model_to_src_model_fixed)

    frame_json['objects'][object_index]['location'] = object_pose.translation.tolist()
    frame_json['objects'][object_index]['quaternion_xyzw'] = np.roll(object_pose.quaternion, -1).tolist()

json.dump(frame_json, args.outfile, indent=2, sort_keys=True)
args.outfile.close()
