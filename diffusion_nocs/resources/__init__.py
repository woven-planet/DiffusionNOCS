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

_resource_dir = pathlib.Path(__file__).parent

DIFFUSION_NOCS_PT_PATH = (
    _resource_dir / "category6.pt"
)  # ["bottle, bowl, camera, can, laptop, mug, unknown"]
DINO_SMALL_MODEL_PATH = _resource_dir / "dinov2-small"
DINO_PATCH_PCA_PATH = _resource_dir / "pca6.pkl"  # 6 dimensions of PCA
IMAGE_BASE_PATH = _resource_dir / "images"
FONT_PATH = _resource_dir / "Arial.ttf"
