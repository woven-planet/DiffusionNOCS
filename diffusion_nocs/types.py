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

from dataclasses import dataclass
from typing import Tuple

CATEGORY_NAME_DICT = {
    "bottle": 0,
    "bowl": 1,
    "camera": 2,
    "can": 3,
    "laptop": 4,
    "mug": 5,
    "unknown": 6,
}
ALL_INPUT_TYPES = ["rgb", "normal", "dino"]


@dataclass
class ImageShape:
    height: int
    width: int
    channel: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.channel)

    def as_torch_tuple(self) -> Tuple[int, int, int]:
        return (self.channel, self.height, self.width)


CROPPED_INPUT_IMAGE_SHAPE = ImageShape(height=160, width=160, channel=12)
