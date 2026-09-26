from typing import List, Sequence, get_args

from facefusion.common_helper import create_float_range, create_int_range
from facefusion.processors.modules.diffusion_swapper.types import DiffusionSwapperModel, DiffusionSwapperSet

diffusion_swapper_set : DiffusionSwapperSet =\
{
	'sd_1_5_inpainting': [ '512x512', '768x768' ]
}

diffusion_swapper_models : List[DiffusionSwapperModel] = list(get_args(DiffusionSwapperModel))

diffusion_swapper_strength_range : Sequence[float] = create_float_range(0.1, 1.0, 0.05)
diffusion_swapper_steps_range : Sequence[int] = create_int_range(8, 64, 1)
diffusion_swapper_scale_range : Sequence[float] = create_float_range(1.0, 10.0, 0.5)
diffusion_swapper_seed_range : Sequence[int] = create_int_range(-1, 2**31 - 1, 1)
