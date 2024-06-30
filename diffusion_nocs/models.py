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
import pickle
from typing import List, Tuple

import cv2
import numpy as np
import torch
from diffusers import DPMSolverSinglestepScheduler, ModelMixin, UNet2DModel
from diffusers.utils import randn_tensor
from sklearn.preprocessing import minmax_scale
from torchvision import transforms
from transformers import AutoModel

from diffusion_nocs.types import CROPPED_INPUT_IMAGE_SHAPE


def create_random_noises(noise_num: int, device: str = "cuda") -> torch.Tensor:
    noise_shape = (
        noise_num,
        3,
    ) + CROPPED_INPUT_IMAGE_SHAPE.as_torch_tuple()[1:]
    seed = 0
    generator = torch.manual_seed(seed)
    noises: torch.Tensor = randn_tensor(noise_shape, generator=generator).to(device)
    return noises


def _build_unet_model() -> ModelMixin:
    height, width, channel = CROPPED_INPUT_IMAGE_SHAPE.as_tuple()
    assert height == width
    model = UNet2DModel(
        sample_size=height,
        in_channels=channel + 3,
        out_channels=3,
        num_class_embeds=7,  # for NOCS
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


class DiffusionInference:
    def __init__(
        self,
        checkpoint_path: pathlib.Path,
        device: str = "cuda",
        timesteps: int = 10,
    ) -> None:
        self._device = device

        self._model = _build_unet_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)  # type: ignore
        self._model.load_state_dict(checkpoint["state_dict"], strict=False)
        self._model.eval()
        self._model.to(device)
        self._scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=1000)
        self._scheduler.set_timesteps(timesteps)

    def predict(self, images: torch.Tensor, cls: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert images.shape[0] == cls.shape[0]
        batch_size = images.shape[0]
        encoder_hidden_states = cls.reshape(batch_size)
        inputs = torch.concat((images, noise), dim=1)

        for t in self._scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = self._model(inputs, t, encoder_hidden_states).sample
                prev_noisy_sample = self._scheduler.step(
                    noisy_residual, t, inputs[:, -3:]
                ).prev_sample
                inputs = torch.concat((images, prev_noisy_sample), dim=1)
        outputs: torch.Tensor = (prev_noisy_sample / 2 + 0.5).clamp(0, 1)
        return outputs


# TODO(taku): conduct refactor
class DinoFeatures:
    def __init__(
        self, model_path: pathlib.Path, pca_path: pathlib.Path, device: str = "cuda"
    ) -> None:
        self._device = device

        self._model = AutoModel.from_pretrained(model_path)
        self.patch_size = self._model.config.patch_size
        self._feature_dim = self._model.config.hidden_size
        self._model.to(device)
        self._model.eval()

        # Normalization transform based on ImageNet
        self._transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        assert pca_path.suffix == ".pkl"
        self._pca = pickle.load(open(str(pca_path), "rb"))

    def get_features(
        self,
        images: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(set(image.shape for image in images)) == 1, "Not all images have same shape"
        im_h, im_w = images[0].shape[0:2]
        assert (
            im_w % self.patch_size == 0 and im_h % self.patch_size == 0
        ), "Both width and height of the image must be divisible by 14"

        image_array = np.stack(images) / 255.0
        input_tensor = torch.Tensor(np.transpose(image_array, [0, 3, 1, 2]))
        input_tensor = self._transform(input_tensor).to(self._device)

        with torch.no_grad():
            outputs = self._model(input_tensor, output_hidden_states=True)
            # CLS token is first then patch tokens
            class_tokens = outputs.last_hidden_state[:, 0, :]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            if patch_tokens.is_cuda:
                patch_tokens = patch_tokens.cpu()
                class_tokens = class_tokens.cpu()
            patch_tokens = patch_tokens.detach().numpy()
            class_tokens = class_tokens.detach().numpy()
            all_patches = patch_tokens.reshape(
                [-1, im_h // self.patch_size, im_w // self.patch_size, self._feature_dim]
            )
        return all_patches, class_tokens

    def apply_pca(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        num_maps, map_w, map_h = features.shape[0:3]
        masked_features = features[masks.astype(bool)]

        # Apply PCA to reduce features
        masked_features_pca = self._pca.transform(masked_features)
        masked_features_pca = minmax_scale(masked_features_pca)

        # Initialize images for pca reduced features
        features_pca_reshaped = np.ones((num_maps, map_w, map_h, masked_features_pca.shape[-1]))

        # Fill in the PCA results only at the masked regions
        features_pca_reshaped[masks.astype(bool)] = masked_features_pca

        return features_pca_reshaped

    def get_pca_features(
        self, rgb_image: np.ndarray, mask: np.ndarray, input_size: int = 448
    ) -> np.ndarray:
        assert mask.dtype == bool
        assert rgb_image.shape[:2] == mask.shape

        patch_size = self.patch_size
        resized_mask = cv2.resize(
            mask * 1,
            (input_size // patch_size, input_size // patch_size),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_rgb = cv2.resize(rgb_image, (input_size, input_size))

        patch_tokens, _ = self.get_features([resized_rgb])
        pca_features = self.apply_pca(patch_tokens, resized_mask[None])

        resized_pca_features: np.ndarray = (
            transforms.functional.resize(
                torch.tensor(pca_features[0]).permute(2, 0, 1),
                size=(160, 160),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            .permute(1, 2, 0)
            .numpy()
        )
        return resized_pca_features
