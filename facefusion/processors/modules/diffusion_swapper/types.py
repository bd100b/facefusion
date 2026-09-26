from typing import Dict, List, Literal, TypeAlias, TypedDict

from facefusion.types import Mask, VisionFrame

DiffusionSwapperInputs = TypedDict('DiffusionSwapperInputs',
{
	'reference_vision_frame' : VisionFrame,
	'source_vision_frames' : List[VisionFrame],
	'target_vision_frames' : List[VisionFrame],
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

DiffusionSwapperModel = Literal['sd_1_5_inpainting']

DiffusionSwapperStrength : TypeAlias = float
DiffusionSwapperSteps : TypeAlias = int
DiffusionSwapperScale : TypeAlias = float
DiffusionSwapperSeed : TypeAlias = int

DiffusionSwapperSet : TypeAlias = Dict[DiffusionSwapperModel, List[str]]
