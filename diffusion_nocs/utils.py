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


import pathlib
from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from diffusion_nocs.resources import FONT_PATH
from diffusion_nocs.types import CATEGORY_NAME_DICT


def build_diffusion_transform() -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


def load_np_rgb_image(file_path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(file_path).convert("RGB"))


def create_image_tensors(
    noises: torch.Tensor,
    rgb_np: np.ndarray,
    normal_np: np.ndarray,
    dino_np: np.ndarray,
    noise_num: int = 1,
    device: str = "cuda",
) -> torch.Tensor:
    diff_trans = build_diffusion_transform()
    rgb_tensor: torch.Tensor = diff_trans(rgb_np)
    dino_tensor: torch.Tensor = diff_trans(dino_np)
    normal_tensor: torch.Tensor = diff_trans(normal_np)
    image_tensor = torch.concat((rgb_tensor, dino_tensor, normal_tensor), dim=0)[None]
    image_tensor = image_tensor.to(device).type_as(noises)
    image_tensors = image_tensor.repeat(noise_num, 1, 1, 1)
    return image_tensors


def create_id_tensors(category_name: str, noise_num: int = 1, device: str = "cuda") -> torch.Tensor:
    id_tensor = torch.Tensor([CATEGORY_NAME_DICT[category_name]]).to(torch.int)
    id_tensor = id_tensor.to(device)
    id_tensors = id_tensor.repeat(noise_num)
    return id_tensors


def render_frame(data: np.ndarray) -> Image.Image:
    data = data.transpose()
    assert data.shape[0] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*data, c=np.transpose(data / 255, (1, 0)))
    ax.view_init(30, 45, vertical_axis="y")
    plt.close()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight", pad_inches=0.1)
    image = Image.open(buf)
    return image


def tile_images(
    images: Sequence[Image.Image], vertical_num: int = 1, horizontal_num: int = 4
) -> Image:
    assert len(images) > 1
    width, height = images[0].size
    canvas = Image.new("RGB", (width * horizontal_num, height * vertical_num))

    for i, image in enumerate(images):
        vertical_count = i // horizontal_num
        horizontal_count = i % horizontal_num

        canvas.paste(image, (width * horizontal_count, height * vertical_count))

    return canvas


def convert_noise_to_image_np(noise_tensor: torch.Tensor) -> np.ndarray:
    image_np: np.ndarray = (
        (noise_tensor / 2 + 0.5).clamp(0, 1).permute((1, 2, 0)).cpu().numpy() * 255
    ).astype(np.uint8)
    return image_np


def convert_nocs256_to_nocs3d(nocs256_np: np.ndarray) -> Image.Image:
    height, width, dim = nocs256_np.shape
    reshaped_nocs256_np = nocs256_np.reshape((height * width, dim))
    point_pil: Image.Image = render_frame(reshaped_nocs256_np).resize((width, height))
    return point_pil


def tile_input_images(
    noise_tensor: torch.Tensor,
    rgb_np: np.ndarray,
    normal_np: np.ndarray,
    dino_np: np.ndarray,
    font_path: pathlib.Path = FONT_PATH,
) -> Image.Image:
    noise_pil = Image.fromarray(convert_noise_to_image_np(noise_tensor))
    rgb_pil = Image.fromarray(rgb_np)
    normal_pil = Image.fromarray(normal_np)
    dino_pil = Image.fromarray((dino_np[..., :3] * 255).astype(np.uint8))
    pil_images = [noise_pil, rgb_pil, normal_pil, dino_pil]
    tiled_pil_image = tile_images(pil_images)

    height, width, offset = 140, 5, 160
    text_locations = [
        (width, height),
        (width + offset, height),
        (width + 2 * offset, height),
        (width + 3 * offset, height),
    ]
    draw = ImageDraw.Draw(tiled_pil_image)
    font = ImageFont.truetype(str(font_path), 17)
    draw.text(
        text_locations[0], "input noise", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    draw.text(
        text_locations[1], "input rgb", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    draw.text(
        text_locations[2], "input normal", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    draw.text(
        text_locations[3], "input dino", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    return tiled_pil_image


def tile_output_images(
    nocs_tensor: torch.Tensor, mask_np: np.ndarray, font_path: pathlib.Path = FONT_PATH
) -> Image.Image:
    pil_trans = transforms.ToPILImage()
    nocs256_np = np.array(pil_trans(nocs_tensor))
    point_pil = convert_nocs256_to_nocs3d(nocs256_np)

    masked_nocs256_np = nocs256_np.copy()
    masked_nocs256_np[np.invert(mask_np)] = 0
    masked_point_pil = convert_nocs256_to_nocs3d(masked_nocs256_np)

    nocs256_pil = Image.fromarray(nocs256_np)
    masked_nocs256_pil = Image.fromarray(masked_nocs256_np)

    pil_images = [nocs256_pil, point_pil, masked_nocs256_pil, masked_point_pil]
    tiled_pil_image = tile_images(pil_images)

    height, width, offset = 140, 5, 160
    text_locations = [
        (width, height),
        (width + offset, height),
        (width + 2 * offset, height),
        (width + 3 * offset, height),
    ]
    draw = ImageDraw.Draw(tiled_pil_image)
    font = ImageFont.truetype(str(font_path), 17)
    draw.text(
        text_locations[0], "output nocs 2d", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    draw.text(
        text_locations[1], "output nocs 3d", "white", font=font, stroke_width=2, stroke_fill="black"
    )
    draw.text(
        text_locations[2],
        "masked output 2d",
        "white",
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    draw.text(
        text_locations[3],
        "masked output 3d",
        "white",
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    return tiled_pil_image
