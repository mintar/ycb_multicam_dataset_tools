ycb_multicam_dataset_tools
==========================

Installation
------------

All instructions have been tested on Ubuntu Xenial 16.04.

**Important:** Make sure that no ROS setup.bash is sourced, otherwise you will
get Python runtime errors from OpenCV / cv2.

```bash
virtualenv -p python2.7 venv_ycbm
source venv_ycbm/bin/activate

pip install meshrender   # requires at least version 0.0.9; in 0.0.6, there is a serious memory leak

git clone https://git.hb.dfki.de/mguenther/ycb_multicam_dataset_tools.git
```

Now adjust the `TOOLS_DIR` and `MESH_DIR` paths in `ycb_multicam_dataset_tools/process_single_folder.sh`.


Usage
-----

Go to the root dir of the YCB-M dataset and run `<path to ycb_multicam_dataset_tools>/process_all.sh`.



Optional: NVidia Dataset Utilities
----------------------------------

Also see: https://github.com/NVIDIA/Dataset_Utilities

The following part is optional. The NVidia Dataset Utilities provide two tools
that are useful in conjunction with the YCB-M dataset:

* `nvdu_viz` for visualizing the dataset
* `nvdu_ycb` for downloading the YCB meshes and transforming them into the
  coordinate system used by the YCB-M and Falling Things dataset. This is not
  necessary, as the models are already contained in the dataset.

To visualize the dataset, follow these instructions (tested on Ubuntu Xenial 16.04): 

```bash
# IMPORTANT: the ROS setup.bash must NOT be sourced, otherwise the following error occurs:
# ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type

# nvdu requires Python 3.5 or 3.6
sudo add-apt-repository -y ppa:deadsnakes/ppa   # to get python3.6 on Ubuntu Xenial
sudo apt-get update
sudo apt-get install -y python3.6 libsm6 libxext6 libxrender1 python-virtualenv python-pip

# create a new virtual environment
virtualenv -p python3.6 venv_nvdu
cd venv_nvdu/
source bin/activate

# clone our fork of NVIDIA's Dataset Utilities that incorporates some essential fixes
pip install -e 'git+https://github.com/mintar/Dataset_Utilities.git#egg=nvdu'

# download and transform the meshes
# (alternatively, unzip the meshes contained in the dataset
# to <path to venv_nvdu>/lib/python3.6/site-packages/nvdu/data/ycb/aligned_cm)
nvdu_ycb -s

# run nvdu_viz to visualize the dataset
cd <a subdirectory of the YCB-M dataset with some frames>
nvdu_viz --name_filters '*.jpg'
```


### How to get aligned_m models (in meters)

Run the following file to convert the `_ycb_original.json` to a `_ycb_aligned_m.json`:

```python
#!/usr/bin/env python

import json

with open('_ycb_original.json', 'r') as f:
    aligned_cm = json.load(f)

for obj in range(len(aligned_cm['exported_objects'])):
    for elem in range(len(aligned_cm['exported_objects'][obj]['cuboid_dimensions'])):
        aligned_cm['exported_objects'][obj]['cuboid_dimensions'][elem] /= 100.0
    for row in range(len(aligned_cm['exported_objects'][obj]['fixed_model_transform'])):
        for elem in range(len(aligned_cm['exported_objects'][obj]['fixed_model_transform'][row])):
            aligned_cm['exported_objects'][obj]['fixed_model_transform'][row][elem] /= 100.0

    aligned_cm['exported_objects'][obj]['fixed_model_transform'][3][3] = 1.0


with open('_ycb_aligned_m.json', 'w') as f:
    json.dump(aligned_cm, f, indent=2)
```

Then rename `_ycb_aligned_m.json` to `_ycb_original.json`, copy it to
lib/python3.6/site-packages/nvdu/config/object_settings/, run nvdu_ycb again
and rename the output dir
(lib/python3.6/site-packages/nvdu/data/ycb/aligned_cm/) from aligned_cm to
aligned_m.
