from .base import REGISTERED_ENVS, MujocoEnv
from .cube_pnp import CubePickAndPlace
from .box_push_cfg import BoxPush
from .push_block_unknown_regions import PushBlockUnknownRegions
from .put_k_exact_blocks import PutKExactBlocks
ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
