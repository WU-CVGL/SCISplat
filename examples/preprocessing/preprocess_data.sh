#!/bin/bash

root_path='/run/determined/workdir/home/gsplat/examples/data/sci_nerf/ablation_study/compression_rate'
scene='factory'
compression_rate='32'


# root_path='/run/determined/workdir/home/gsplat/examples/data/sci_nerf/ablation_study/mask_overlapping_rate'
# scene='airplants125'

# python interpolate_decoded_imgs.py $root_path $scene

/run/determined/workdir/miniconda3/envs/vggsfm/bin/python /run/determined/workdir/home/vggsfm/demo.py \
    SCENE_DIR="${root_path}/${scene}${compression_rate}" \
    query_frame_num=1 \
    shared_camera=true \
    max_query_pts=2048 \
    make_reproj_video=true

# check if the run above has no error, then proceed to this visualization code
# /run/determined/workdir/miniconda3/envs/vggsfm/bin/python /run/determined/workdir/home/vggsfm/colmap_vis.py \
#     --image_dir "${root_path}/${scene}${compression_rate}/images" \
#     --colmap_path "/run/determined/workdir/home/vggsfm/output/${scene}" \
#     --downsample 1
