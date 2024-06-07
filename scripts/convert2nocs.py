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


"""
BOP to NOCS dataset convertion script. 
Parts of the code are borrowed from the BOP toolkit (https://github.com/thodan/bop_toolkit) 
and NOCS (https://github.com/hughw19/NOCS_CVPR2019) GitHub repositories.
"""

import argparse
import glob
import json
import os
import shutil

import _pickle as cPickle
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot


class BOPDataset:
    """BOP dataset class"""

    def __init__(self, path, split="val", scene_ids=None, save_models=True):

        self.path = os.path.join(path, split)
        self.scene_ids = [int(os.path.basename(s)) for s in glob.glob(os.path.join(self.path, "*"))]
        if scene_ids:
            self.scene_ids = scene_ids

        self.db_info = {}
        self.db_info["scene_camera_tpath"] = os.path.join(self.path, "{:06d}", "scene_camera.json")
        self.db_info["scene_gt_tpath"] = os.path.join(self.path, "{:06d}", "scene_gt.json")
        self.db_info["image_tpath"] = os.path.join(self.path, "{:06d}", "{}", "{:06d}.{}")
        self.db_info["mvisib_tpath"] = os.path.join(self.path, "{:06d}", "{}", "{:06d}_{:06d}.{}")
        self.db_info["model_tpath"] = os.path.join(path, "models_eval")

        self.split = split
        self.scenes_gt = {}
        self.scenes_camera = {}
        self.images = []

        # Load scenes and poses
        for scene_id in self.scene_ids:
            # Load scene info and ground-truth poses.
            self.scenes_camera[scene_id] = self._load_scene_camera(
                self.db_info["scene_camera_tpath"].format(scene_id)
            )
            self.scenes_gt[scene_id] = self._load_scene_gt(
                self.db_info["scene_gt_tpath"].format(scene_id)
            )

        # Collect images
        for scene_id in self.scene_ids:
            for image_id in self.scenes_gt[scene_id].keys():
                self.images.append(
                    self.db_info["image_tpath"].format(scene_id, "rgb", image_id, "png")
                )

        # Object ID / Class associations
        self.class_to_ids = {}
        if "hope" in path:
            self.class_to_ids["bottle"] = [2, 11, 13, 16, 24]
            self.class_to_ids["can"] = [1, 4, 7, 10, 15, 18, 19, 20, 21, 26, 27, 28]
            self.class_to_ids["bowl"] = []
            self.class_to_ids["mug"] = []
        elif "ycbv" in path:
            self.class_to_ids["bottle"] = [5, 12]
            self.class_to_ids["can"] = [1, 4, 6]
            self.class_to_ids["bowl"] = [13]
            self.class_to_ids["mug"] = [14]
        elif "tyol" in path:
            self.class_to_ids["bottle"] = [8]
            self.class_to_ids["can"] = [10, 11]
            self.class_to_ids["bowl"] = [3, 4, 5, 6, 7, 20, 21]
            self.class_to_ids["mug"] = [17, 18]

    def save_as_nocs(self, output_dir, save_models=True):
        """Converts and saves a BOP dataset in NOCS format"""
        for idx, rgb_path in enumerate(self.images):
            image_id, scene_id = self._extract_ids(self.images[idx])

            # Read camera parameters and instance masks
            K = self.scenes_camera[scene_id][image_id]["cam_K"]
            depth_path = self.db_info["image_tpath"].format(scene_id, "depth", image_id, "png")
            mask_visib = self._load_masks(
                self.images[idx][:-4].replace("rgb", "{}"), image_id, mask_type="mask_visib"
            )

            # Read image height and width
            img = cv2.imread(rgb_path)
            img_height, img_width = img.shape[:2]
            meta_tuple_list = []

            # Store R,t,s and convert instance masks to NOCS format
            counter = 1
            all_valid_masks = []
            all_rotations = []
            all_translations = []
            all_scales = []
            all_gt_poses = []
            for gt_id, gt in enumerate(self.scenes_gt[scene_id][image_id]):
                obj_id = gt["obj_id"]

                rotation = self._read_gt_pose(gt)[:3, :3]
                translation = self._read_gt_pose(gt)[:, -1:] * 0.001  # Scale to meters

                # Correct rotation
                R_corr = Rot.from_euler("x", 90, degrees=True).as_matrix()
                R_corr_1 = np.eye(3)
                if obj_id == 12:
                    R_corr_1 = Rot.from_euler("y", 90, degrees=True).as_matrix()
                if obj_id == 5:
                    R_corr_1 = Rot.from_euler("y", 67, degrees=True).as_matrix()
                rotation = rotation @ R_corr @ R_corr_1

                if (
                    obj_id in self.class_to_ids["bottle"]
                    or obj_id in self.class_to_ids["can"]
                    or obj_id in self.class_to_ids["bowl"]
                    or obj_id in self.class_to_ids["mug"]
                ):
                    mask = mask_visib[gt_id].clip(0, 1)
                    mask[mask == 1] = counter
                    all_valid_masks.append(mask)

                    if obj_id in self.class_to_ids["bottle"]:
                        meta_tuple_list.append((counter, 1, obj_id))
                    elif obj_id in self.class_to_ids["can"]:
                        meta_tuple_list.append((counter, 4, obj_id))
                    elif obj_id in self.class_to_ids["bowl"]:
                        meta_tuple_list.append((counter, 2, obj_id))
                    elif obj_id in self.class_to_ids["mug"]:
                        meta_tuple_list.append((counter, 6, obj_id))

                    all_scales.append(1)
                    all_rotations.append(rotation)
                    all_translations.append(translation)

                    camera_T_object = np.eye(4)
                    camera_T_object[:3, :3] = rotation
                    camera_T_object[:3, 3] = np.squeeze(translation, axis=1)
                    all_gt_poses.append(camera_T_object)
                    counter += 1

            # Skip frame if no objects of interest found
            if len(all_rotations) == 0:
                continue

            combined_mask = np.zeros((img_height, img_width), dtype=np.int32)
            all_2D_boxes = []

            # Iterate through each mask and assign unique labels
            for idx, mask in enumerate(all_valid_masks, start=1):
                # Add 1 to each pixel value in the mask to avoid overlap with existing labels
                mask = idx * (mask > 0)

                # Find the bounding box of the mask
                pos = np.where(mask > 0)
                y1, x1 = np.min(pos, axis=1)
                y2, x2 = np.max(pos, axis=1)
                all_2D_boxes.append([y1, x1, y2, x2])

                # Combine the masks
                combined_mask = np.maximum(combined_mask, mask)

            output_dir_scene = os.path.join(output_dir, "val", f"{scene_id:06d}")
            os.makedirs(output_dir_scene, exist_ok=True)
            output_rgb_path = os.path.join(output_dir_scene, f"{image_id:06d}_color.png")
            output_depth_path = os.path.join(output_dir_scene, f"{image_id:06d}_depth.png")

            with open(os.path.join(output_dir_scene, f"{image_id:06d}_meta.txt"), "w") as f:
                for meta_tuple in meta_tuple_list:
                    f.write(
                        str(meta_tuple[0])
                        + " "
                        + str(meta_tuple[1])
                        + " "
                        + str(meta_tuple[2])
                        + "\n"
                    )

            # Save combined mask
            cv2.imwrite(
                os.path.join(output_dir_scene, f"{image_id:06d}_mask.png"),
                combined_mask.astype(np.uint8),
            )

            # Save RGB and Depth
            shutil.copy(rgb_path, output_rgb_path)
            depth_scale = self.scenes_camera[scene_id][image_id]["depth_scale"]
            if depth_scale == 1:
                shutil.copy(depth_path, output_depth_path)
            else:
                depth_tmp = cv2.imread(depth_path, -1) * depth_scale
                cv2.imwrite(output_depth_path, depth_tmp.astype(np.uint16))

            all_rotations = np.array(all_rotations)
            all_translations = np.array(all_translations)
            all_scales = np.array(all_scales)
            all_gt_poses = np.array(all_gt_poses)

            class_ids = np.array([x[1] for x in meta_tuple_list])
            instance_ids = [x[0] for x in meta_tuple_list]
            model_list = [x[2] for x in meta_tuple_list]

            gts = {}
            gts["class_ids"] = class_ids  # int list, 1 to 6
            gts["bboxes"] = all_2D_boxes  # np.array, [[y1, x1, y2, x2], ...]
            gts["scales"] = all_scales.astype(
                np.float32
            )  # np.array, scale factor from NOCS model to depth observation
            gts["poses"] = all_gt_poses.astype(np.float32)  # np.array, [R|T], 3x4 matrix
            gts["rotations"] = all_rotations.astype(np.float32)  # np.array, R
            gts["translations"] = all_translations.astype(np.float32)  # np.array, T
            gts["instance_ids"] = instance_ids  # int list, start from 1
            gts["model_list"] = model_list  # str list, model id/name

            # Save labels
            with open(os.path.join(output_dir_scene, f"{image_id:06d}_label.pkl"), "wb") as f:
                cPickle.dump(gts, f)

            # Save camera intrinsics
            with open(os.path.join(output_dir_scene, f"{image_id:06d}_camK.txt"), "w") as f:
                for row in K:
                    f.write(" ".join([str(x) for x in row]) + "\n")

        # Save image list
        self.create_data_list(output_dir, "val")
        # Save mesh models
        if save_models:
            self.save_models(output_dir)

    def save_models(self, output_dir):
        """Saves corrected mesh models"""
        import open3d as o3d

        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        objects = sorted(glob.glob(os.path.join(self.db_info["model_tpath"], "*.ply")))
        for i, o in enumerate(objects):
            obj_id = int(os.path.basename(o)[4:-4])
            mesh = o3d.io.read_triangle_mesh(
                os.path.join(self.db_info["model_tpath"], f"obj_{obj_id:06d}.ply")
            )
            mesh.scale(0.001, center=(0, 0, 0))
            R_corr = Rot.from_euler("x", -90, degrees=True).as_matrix()
            R_corr_1 = np.eye(3)
            if obj_id == 12:
                R_corr_1 = Rot.from_euler("y", -90, degrees=True).as_matrix()
            elif obj_id == 5:
                R_corr_1 = Rot.from_euler("y", -67, degrees=True).as_matrix()

            mesh.rotate(R_corr_1 @ R_corr, center=(0, 0, 0))
            o3d.io.write_triangle_mesh(os.path.join(models_dir, f"obj_{obj_id:06d}.ply"), mesh)

    def create_data_list(self, data_dir, subset):
        """Creates a file list in the NOCS format"""
        img_dir = os.path.join(data_dir, subset)
        folder_list = [
            name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))
        ]
        all_image_paths = []
        for i in range(len(folder_list)):
            image_paths = glob.glob(os.path.join(img_dir, folder_list[i], "*_color.png"))
            all_image_paths.extend(image_paths)

        with open(os.path.join(data_dir, subset + "_list_all.txt"), "w") as f:
            for img_path in all_image_paths:
                # Join the parts starting from 'val' till the end of the path
                extracted_path = os.path.join(
                    *img_path.split("/")[img_path.split("/").index("val") :]
                ).replace("_color.png", "")
                f.write("%s\n" % extracted_path)

    def _read_gt_pose(self, gt_img):
        """Reads ground truth rotation and translation"""
        rotation = np.array(gt_img["cam_R_m2c"]).reshape((3, 3))
        translation = np.array(gt_img["cam_t_m2c"]).reshape((3,))

        gt_pose = np.empty((3, 4))
        gt_pose[:3, :3] = rotation
        gt_pose[:3, 3] = translation

        return gt_pose

    def _load_scene_camera(self, path):
        """Loads content of a JSON file with information about the scene camera."""
        scene_camera = self._load_json(path, keys_to_int=True)

        for im_id in scene_camera.keys():
            if "cam_K" in scene_camera[im_id].keys():
                scene_camera[im_id]["cam_K"] = np.array(
                    scene_camera[im_id]["cam_K"], float
                ).reshape((3, 3))
            if "cam_R_w2c" in scene_camera[im_id].keys():
                scene_camera[im_id]["cam_R_w2c"] = np.array(
                    scene_camera[im_id]["cam_R_w2c"], float
                ).reshape((3, 3))
            if "cam_t_w2c" in scene_camera[im_id].keys():
                scene_camera[im_id]["cam_t_w2c"] = np.array(
                    scene_camera[im_id]["cam_t_w2c"], float
                ).reshape((3, 1))
        return scene_camera

    def _load_scene_gt(self, path):
        """Loads content of a JSON file with ground-truth annotations."""
        scene_gt = self._load_json(path, keys_to_int=True)

        for im_id, im_gt in scene_gt.items():
            for gt in im_gt:
                if "cam_R_m2c" in gt.keys():
                    gt["cam_R_m2c"] = np.array(gt["cam_R_m2c"], float).reshape((3, 3))
                if "cam_t_m2c" in gt.keys():
                    gt["cam_t_m2c"] = np.array(gt["cam_t_m2c"], float).reshape((3, 1))
        return scene_gt

    def _load_json(self, path, keys_to_int=False):
        """Loads content of a JSON file."""

        # Keys to integers.
        def convert_keys_to_int(x):
            return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

        with open(path, "r") as f:
            if keys_to_int:
                content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
            else:
                content = json.load(f)

        return content

    def _extract_ids(self, path):
        """Extracts scene and image ids from the file name"""
        image_id = int(os.path.basename(path)[:-4])
        scene_id = int(os.path.basename(os.path.dirname(os.path.dirname(path))))
        return image_id, scene_id

    def _load_masks(
        self, scene_dir, image_id, mask_type="mask", n_instances=None, instance_ids=None
    ):
        """Loads one or multiple object masks stored as individual .png files."""

        if n_instances is not None and instance_ids is None:
            instance_ids = range(n_instances)

        if instance_ids is not None:
            mask_paths = (
                scene_dir / mask_type / f"{image_id:06d}_{instance_id:06d}.png"
                for instance_id in instance_ids
            )
        else:
            mask_paths = glob.glob(scene_dir.format(mask_type) + "_*.png")
            mask_paths = sorted(mask_paths, key=lambda p: int(p.split("_")[-1][:-4]))
        masks = np.stack([np.array(Image.open(p)) for p in mask_paths])
        return masks


def main():
    # Parse input
    parser = argparse.ArgumentParser(
        "Convert BOP dataset (https://bop.felk.cvut.cz/datasets/) to NOCS format"
    )
    parser.add_argument("--data_dir", default="data/hope")
    parser.add_argument("--output_dir", default="data/hope_nocs")
    parser.add_argument("--data_split", default="val")
    parser.add_argument("--save_models", action="store_true", default=False)
    args = parser.parse_args()

    # Process data
    dataset = BOPDataset(args.data_dir, split=args.data_split)
    dataset.save_as_nocs(args.output_dir, save_models=args.save_models)


if __name__ == "__main__":
    main()
