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
import pathlib

import numpy as np

from diffusion_nocs.models import DiffusionInference, DinoFeatures, create_random_noises
from diffusion_nocs.resources import (
    DIFFUSION_NOCS_PT_PATH,
    DINO_PATCH_PCA_PATH,
    DINO_SMALL_MODEL_PATH,
    IMAGE_BASE_PATH,
)
from diffusion_nocs.types import ALL_INPUT_TYPES, CATEGORY_NAME_DICT
from diffusion_nocs.utils import (
    create_id_tensors,
    create_image_tensors,
    load_np_rgb_image,
    tile_input_images,
    tile_output_images,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=IMAGE_BASE_PATH,
        help="Directory of input images.",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="bottle",
        choices=[*CATEGORY_NAME_DICT],
        help="One from bottle, bowl, camera, can, laptop, mug, unknown",
    )
    parser.add_argument("--input-types", nargs="+", default=ALL_INPUT_TYPES)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device",
    )
    parser.add_argument(
        "--noise-num",
        type=int,
        default=2,
        help="Number of noise",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Denoising iterations",
    )
    return parser


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    device = args.device

    base_dir = args.input_dir
    rgb_np = load_np_rgb_image(base_dir / f"{args.category_name}" / "rgb.png")
    normal_np = load_np_rgb_image(base_dir / f"{args.category_name}" / "normal.png")

    # NOTE: we assume the background of RGB image is white
    mask_np = rgb_np.sum(-1) < (255 * 3)

    dinov2 = DinoFeatures(DINO_SMALL_MODEL_PATH, DINO_PATCH_PCA_PATH)
    dino_np = dinov2.get_pca_features(rgb_np, mask_np)

    noise_num = args.noise_num
    noises = create_random_noises(noise_num, args.device)

    if "rgb" not in args.input_types:
        rgb_np = np.zeros_like(rgb_np)
    if "normal" not in args.input_types:
        normal_np = np.zeros_like(normal_np)
    if "dino" not in args.input_types:
        dino_np = np.zeros_like(dino_np)

    tile_input_image = tile_input_images(noises[0], rgb_np, normal_np, dino_np)
    tile_input_image.show()

    image_tensors = create_image_tensors(noises, rgb_np, normal_np, dino_np, noise_num, device)
    id_tensors = create_id_tensors(args.category_name, noise_num, device)

    diffusion_nocs = DiffusionInference(DIFFUSION_NOCS_PT_PATH, timesteps=args.timesteps)
    nocs_tensors = diffusion_nocs.predict(image_tensors, id_tensors, noises)

    for nocs_tensor in nocs_tensors:
        tile_output_image = tile_output_images(nocs_tensor, mask_np)
        tile_output_image.show()


if __name__ == "__main__":
    main()
