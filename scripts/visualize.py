# Copyright 2024 TOYOTA MOTOR CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

import _pickle as cPickle
import numpy as np
import open3d as o3d

# Category ID to category name dictionary
catId_to_name = {0: "bottle", 1: "bowl", 2: "camera", 3: "can", 4: "laptop", 5: "mug"}


def visualize(data_dir, num_frames=10):
    """Visualizes N frames from the selected dataset in NOCS format"""
    test_img_list_path = os.path.join(data_dir, "val_list_all.txt")
    img_list = open(test_img_list_path).read().splitlines()
    step = int((len(img_list) / num_frames))
    img_list = img_list[::step]

    # Loop through image list
    for img_idx in img_list:
        # Load images and annotations
        all_pcds = []
        img_path = os.path.join(data_dir, img_idx)
        K = np.loadtxt(img_path + "_camK.txt")
        rgb = o3d.io.read_image(img_path + "_color.png")
        mask = o3d.io.read_image(img_path + "_mask.png")
        depth = o3d.io.read_image(img_path + "_depth.png")

        rgb = o3d.geometry.Image(
            (
                np.array(rgb).astype(float)
                * (np.array(mask)[..., None].astype(bool).astype(float) + 0.3).clip(0, 1)
            ).astype(np.uint8)
        )

        im_H, im_W = np.array(rgb).shape[0], np.array(rgb).shape[1]
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, convert_rgb_to_intensity=False, depth_trunc=0.9
        )
        K_o3d = o3d.camera.PinholeCameraIntrinsic(im_H, im_W, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        pcd_depth = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, K_o3d)
        all_pcds.append(pcd_depth)

        with open(img_path + "_label.pkl", "rb") as f:
            gts = cPickle.load(f)

        # Load and transform meshes
        model_path = os.path.join(data_dir, "models")
        print("===== Scene {} =====".format(int(img_idx.split("/")[1])))
        for idx in range(len(gts["instance_ids"])):
            # Load mesh model
            inst_id = gts["instance_ids"][idx]
            cat_id = gts["class_ids"][idx] - 1  # convert to 0-indexed
            model_name = gts["model_list"][idx]
            model_load_path = os.path.join(model_path, f"obj_{model_name:06d}.ply")
            mesh_model = o3d.io.read_triangle_mesh(model_load_path)

            # Transform mesh model
            camera_T_object = np.eye(4)
            camera_T_object[:3, :3] = gts["rotations"][idx]
            camera_T_object[:3, 3] = gts["translations"][idx].flatten()
            mesh_model.transform(camera_T_object)
            mesh_model.compute_vertex_normals()
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )
            coordinate_frame.transform(camera_T_object)

            all_pcds.append(mesh_model)
            all_pcds.append(coordinate_frame)

            print("Obj ID: {}, Class: {}".format(model_name, catId_to_name[cat_id]))

        # Visualize scene
        o3d.visualization.draw_geometries(all_pcds, width=800, height=800)


def main():
    # Parse input
    parser = argparse.ArgumentParser("Visualize converted BOP datasets")
    parser.add_argument("--data_dir", default="datasets/hope")
    args = parser.parse_args()

    # Process data
    visualize(args.data_dir)


if __name__ == "__main__":
    main()
