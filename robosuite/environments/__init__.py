from .base import REGISTERED_ENVS, MujocoEnv
from .cdp.cube_pnp import CubePickAndPlace
from .cdp.box_push_cdp import BoxPush
from .pomdp.push_block_unknown_regions import PushBlockUnknownRegions
from .pomdp.put_k_exact_blocks import PutKExactBlocks
from .pomdp.button_lightbulb import ButtonLightbulb
from .pomdp.fruit_swap import FruitSwap
from .pomdp.fruit_swap_vision import FruitSwapVision
from .cdp.cube_pnp_cdp import CubePlaceCDP
from .cdp.cube_pnp_cdp_vision import CubePlaceCDPVision
from .cdp.cube_prp_cdp import CubePickRotatePlaceCDP
ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
