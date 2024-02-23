# https://github.com/mihaidusmanu/d2-net/blob/master/megadepth_utils/undistort_reconstructions.py
""" 
Undistort the MegaDepth reconstructions using COLMAP and update the reconstructions.
The results will be used to calculate camera poses that will be used to train LoFTR.
You need to have COLMAP installed and the COLMAP executables in your PATH.
Download the MegaDepth dataset from https://www.cs.cornell.edu/projects/megadepth/. 
The MegaDepth dataset should be organized as follows:
- base_path
    - MegaDepth_v1_SfM
    - phoenix
        - S6
            - zl548
                - MegaDepth_v1
                    - scene_name
                        - dense0
                            - imgs
                            - depth
"""

import os
import subprocess
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(
    "Undistort the MegaDepth reconstructions using COLMAP."
)
parser.add_argument(
    "--colmap_path",
    type=str,
    default="/usr/local/bin",
    help="Path to the COLMAP executables.",
)
parser.add_argument(
    "--base_path",
    type=str,
    help="Path to the MegaDepth dataset.",
)

args = parser.parse_args()
colmap_path = args.colmap_path
base_path = args.base_path
sfm_path = os.path.join(base_path, "MegaDepth_v1_SfM")
base_depth_path = os.path.join(base_path, "phoenix/S6/zl548/MegaDepth_v1")
output_path = os.path.join(base_path, "Undistorted_SfM")

os.makedirs(output_path, exist_ok=True)

for scene_name in tqdm(os.listdir(base_depth_path)):
    print(f"Processing {scene_name}...")
    current_output_path = os.path.join(output_path, scene_name)
    os.makedirs(current_output_path, exist_ok=True)

    output_image_dir = os.path.join(current_output_path, 'images')
    os.makedirs(output_image_dir, exist_ok=True)

    image_path = os.path.join(base_depth_path, scene_name, "dense0", "imgs")
    if not os.path.exists(image_path):
        continue

    if len(os.listdir(image_path)) == len(os.listdir(output_image_dir)):
        print(f"Skip {scene_name} as it has been processed.") 
        continue

    # Find the maximum image size in scene.
    max_image_size = 0
    for image_name in os.listdir(image_path):
        im = Image.open(os.path.join(image_path, image_name))
        max_image_size = max(max_image_size, max(im.size))

    # Undistort the images and update the reconstruction.
    subprocess.call(
        [
            os.path.join(colmap_path, "colmap"),
            "image_undistorter",
            "--image_path",
            os.path.join(sfm_path, scene_name, "images"),
            "--input_path",
            os.path.join(sfm_path, scene_name, "sparse", "manhattan", "0"),
            "--output_path",
            current_output_path,
            "--max_image_size",
            str(max_image_size),
        ]
    )

    # Transform the reconstruction to raw text format.
    sparse_txt_path = os.path.join(current_output_path, "sparse-txt")
    os.makedirs(sparse_txt_path, exist_ok=True)
    subprocess.call(
        [
            os.path.join(colmap_path, "colmap"),
            "model_converter",
            "--input_path",
            os.path.join(current_output_path, "sparse"),
            "--output_path",
            sparse_txt_path,
            "--output_type",
            "TXT",
        ]
    )
