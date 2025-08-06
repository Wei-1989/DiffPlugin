# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

from MyHook import CrossAttnDownBlock2D_hook_fn

from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    get_down_block,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from Myutils import visualization_tensor


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

@dataclass
class SCBNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name