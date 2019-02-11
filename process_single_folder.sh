#!/bin/bash

TOOLS_DIR=/home/martin/ros/kinetic/segmentation_imgs/src/ycb_multicam_dataset_tools
MESH_DIR=/home/martin/Downloads/ycb/dope_models/aligned_cm/
PATH=$TOOLS_DIR:$PATH

cp $TOOLS_DIR/config/aligned_cm_object_settings.json _object_settings.json

make_segmentation_imgs.py --data-dir . \
                          --mesh-dir $MESH_DIR \
                          --object-settings $TOOLS_DIR/config/ycb_object_settings.json \
                          --target-object-settings $TOOLS_DIR/config/aligned_m_object_settings.json

for f in *.ycbm_full.json; do
  PREFIX="$(basename $f .ycbm_full.json)"
  if [ -f "$PREFIX.intensity.jpg" ]; then OUTNAME="$PREFIX.intensity.json"; fi
  if [ -f "$PREFIX.rgb.jpg" ]; then OUTNAME="$PREFIX.rgb.json"; fi
  if [ -z $OUTNAME ]; then
    echo "ERROR: No such file: $PREFIX.*.jpg!"
    exit 1
  fi
  convert_json_units.py --unit-scaling 100 "$PREFIX.ycbm_full.json" "$OUTNAME"
done

