from argparse import ArgumentParser
from functools import lru_cache
from types import ModuleType
from typing import Any, Dict, List, Tuple

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager, voice_extractor
from facefusion.common_helper import get_middle
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_creator import scale_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_selector import select_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors.modules.diffusion_swapper import choices as diffusion_swapper_choices
from facefusion.processors.modules.diffusion_swapper.types import DiffusionSwapperInputs
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadSet, Face, InferencePool, Mask, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import conditional_match_frame_color, read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadSet) -> ModelSet: #type:ignore[arg-type]
	return\
	{
		'sd_1_5_inpainting':
		{
			'__metadata__':
			{
				'vendor': 'RunwayML',
				'license': 'CreativeML OpenRAIL-M',
				'year': 2022
			},
			'hashes':
			{
				'unet':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_unet.hash'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_unet.hash')
				},
				'vae_encoder':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_vae_encoder.hash'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_vae_encoder.hash')
				},
				'vae_decoder':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_vae_decoder.hash'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_vae_decoder.hash')
				}
			},
			'sources':
			{
				'unet':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_unet.onnx'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_unet.onnx')
				},
				'vae_encoder':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_vae_encoder.onnx'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_vae_encoder.onnx')
				},
				'vae_decoder':
				{
					'url': resolve_download_url('diffusion-3.0.0', 'sd_1_5_inpainting_vae_decoder.onnx'),
					'path': resolve_relative_path('../.assets/models/sd_1_5_inpainting/sd_1_5_inpainting_vae_decoder.onnx')
				}
			},
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ 'sd_1_5_inpainting_unet', 'sd_1_5_inpainting_vae_encoder', 'sd_1_5_inpainting_vae_decoder' ]
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ 'sd_1_5_inpainting_unet', 'sd_1_5_inpainting_vae_encoder', 'sd_1_5_inpainting_vae_decoder' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def collect_model_downloads() -> Tuple[Dict[str, Any], DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set : Dict[str, Any] = {}
	model_source_set : DownloadSet = {}

	for download_name, download in model_set.get('sd_1_5_inpainting').get('hashes').items():
		model_hash_set['sd_1_5_inpainting_' + download_name] = download

	for download_name, download in model_set.get('sd_1_5_inpainting').get('sources').items():
		model_source_set['sd_1_5_inpainting_' + download_name] = download

	return model_hash_set, model_source_set


def get_common_modules() -> List[ModuleType]:
	return [ content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, voice_extractor ]


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()

	for common_module in get_common_modules():
		if not common_module.pre_check():
			return False

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--diffusion-swapper-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'diffusion_swapper_model', 'sd_1_5_inpainting'), choices = diffusion_swapper_choices.diffusion_swapper_models)
		group_processors.add_argument('--diffusion-swapper-strength', help = translator.get('help.strength', __package__), type = float, default = config.get_float_value('processors', 'diffusion_swapper_strength', '0.55'), choices = diffusion_swapper_choices.diffusion_swapper_strength_range)
		group_processors.add_argument('--diffusion-swapper-steps', help = translator.get('help.steps', __package__), type = int, default = config.get_int_value('processors', 'diffusion_swapper_steps', '24'), choices = diffusion_swapper_choices.diffusion_swapper_steps_range)
		group_processors.add_argument('--diffusion-swapper-scale', help = translator.get('help.scale', __package__), type = float, default = config.get_float_value('processors', 'diffusion_swapper_scale', '4.5'), choices = diffusion_swapper_choices.diffusion_swapper_scale_range)
		group_processors.add_argument('--diffusion-swapper-seed', help = translator.get('help.seed', __package__), type = int, default = config.get_int_value('processors', 'diffusion_swapper_seed', '-1'), choices = diffusion_swapper_choices.diffusion_swapper_seed_range)
		facefusion.jobs.job_store.register_step_keys([ 'diffusion_swapper_model', 'diffusion_swapper_strength', 'diffusion_swapper_steps', 'diffusion_swapper_scale', 'diffusion_swapper_seed' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('diffusion_swapper_model', args.get('diffusion_swapper_model'))
	apply_state_item('diffusion_swapper_strength', args.get('diffusion_swapper_strength'))
	apply_state_item('diffusion_swapper_steps', args.get('diffusion_swapper_steps'))
	apply_state_item('diffusion_swapper_scale', args.get('diffusion_swapper_scale'))
	apply_state_item('diffusion_swapper_seed', args.get('diffusion_swapper_seed'))


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()

	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()

	if state_manager.get_item('video_memory_strategy') == 'strict':
		for common_module in get_common_modules():
			common_module.clear_inference_pool()


def swap_face(target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_size = get_model_options().get('size')
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'arcface_128', model_size)
	crop_mask = create_crop_mask(crop_vision_frame, target_face, affine_matrix)
	crop_vision_frame = generate_crop_frame(crop_vision_frame)
	crop_vision_frame = conditional_match_frame_color(temp_vision_frame, crop_vision_frame)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def create_crop_mask(crop_vision_frame : VisionFrame, target_face : Face, affine_matrix : Any) -> Mask:
	crop_masks = []
	padding = state_manager.get_item('face_mask_padding')

	if 'box' in state_manager.get_item('face_mask_types'):
		box_mask = face_masker.create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), padding)
		crop_masks.append(box_mask)

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = face_masker.create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	if 'area' in state_manager.get_item('face_mask_types'):
		face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		area_mask = face_masker.create_area_mask(crop_vision_frame, face_landmark_68, state_manager.get_item('face_mask_areas'))
		crop_masks.append(area_mask)

	if 'region' in state_manager.get_item('face_mask_types'):
		region_mask = face_masker.create_region_mask(crop_vision_frame, state_manager.get_item('face_mask_regions'))
		crop_masks.append(region_mask)

	if '3d' in state_manager.get_item('face_mask_types'):
		face_landmark_5_crop = cv2.transform(target_face.landmark_set.get('5').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		mask_3d = face_masker.create_3d_mask(crop_vision_frame, face_landmark_5_crop, state_manager.get_item('face_mask_blur'), padding)
		crop_masks.append(mask_3d)

	if not crop_masks:
		box_mask = face_masker.create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), padding)
		crop_masks.append(box_mask)

	return numpy.minimum.reduce(crop_masks).clip(0, 1)


def generate_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	strength = state_manager.get_item('diffusion_swapper_strength')
	steps = state_manager.get_item('diffusion_swapper_steps')
	identity_scale = (state_manager.get_item('diffusion_swapper_scale') - 1) * 0.05
	seed = state_manager.get_item('diffusion_swapper_seed')

	rng = numpy.random.default_rng(seed if seed >= 0 else None)
	alphas_cumprod = create_alphas_cumprod(1000)
	init_timestep = max(int(1000 * strength), 1)
	timestep_set = numpy.linspace(init_timestep, 0, steps + 1).astype(numpy.int64)

	latents_target = forward_encode_pixels(crop_vision_frame)
	init_noise = rng.standard_normal(latents_target.shape).astype(numpy.float32)
	latents = numpy.sqrt(alphas_cumprod[init_timestep]) * latents_target + numpy.sqrt(1 - alphas_cumprod[init_timestep]) * init_noise

	with thread_semaphore():
		for timestep, next_timestep in zip(timestep_set[:-1], timestep_set[1:]):
			noise_pred = forward_denoise_frame(latents, int(timestep))
			alpha_prod = alphas_cumprod[int(timestep)]
			alpha_prod_next = alphas_cumprod[int(next_timestep)]
			latents_0 = (latents - numpy.sqrt(1 - alpha_prod) * noise_pred) / numpy.sqrt(alpha_prod)
			latents_0 = latents_0 * (1 - identity_scale) + latents_target * identity_scale

			if next_timestep > 0:
				latents = numpy.sqrt(alpha_prod_next) * latents_0 + numpy.sqrt(1 - alpha_prod_next) * noise_pred
			else:
				latents = latents_0

	return forward_decode_pixels(latents)


def create_alphas_cumprod(timestep_total : int) -> VisionFrame:
	# linear beta schedule of stable diffusion v1
	beta_start = 0.00085
	beta_end = 0.012
	betas = numpy.linspace(beta_start, beta_end, timestep_total, dtype = numpy.float64)
	return numpy.cumprod(1 - betas)


def forward_encode_pixels(crop_vision_frame : VisionFrame) -> VisionFrame:
	vae_encoder = get_inference_pool().get('sd_1_5_inpainting_vae_encoder')
	prepare_vision_frame = crop_vision_frame[:, :, ::-1].astype(numpy.float32) / 127.5 - 1.0
	prepare_vision_frame = prepare_vision_frame.transpose(2, 0, 1)
	prepare_vision_frame = numpy.expand_dims(prepare_vision_frame, axis = 0)

	with thread_semaphore():
		latents = vae_encoder.run(None, { vae_encoder.get_inputs()[0].name: prepare_vision_frame })[0][0]

	return latents * 0.18215


def forward_decode_pixels(latents : VisionFrame) -> VisionFrame:
	vae_decoder = get_inference_pool().get('sd_1_5_inpainting_vae_decoder')
	prepare_latents = latents / 0.18215
	prepare_latents = numpy.expand_dims(prepare_latents, axis = 0)

	with thread_semaphore():
		pixels = vae_decoder.run(None, { vae_decoder.get_inputs()[0].name: prepare_latents })[0][0]

	pixels = pixels.clip(-1, 1)
	pixels = ((pixels + 1) * 127.5).astype(numpy.uint8)
	return pixels.transpose(1, 2, 0)


def forward_denoise_frame(latents : VisionFrame, timestep : int) -> VisionFrame:
	unet = get_inference_pool().get('sd_1_5_inpainting_unet')
	prepare_latents = numpy.expand_dims(latents, axis = 0).astype(numpy.float32)
	timestep_input = numpy.array([ timestep ], dtype = numpy.float32)
	unet_inputs = {}

	for unet_input in unet.get_inputs():
		if unet_input.name == 'sample':
			unet_inputs[unet_input.name] = prepare_latents
		if unet_input.name == 'timestep':
			unet_inputs[unet_input.name] = timestep_input
		if unet_input.name == 'encoder_hidden_states':
			unet_inputs[unet_input.name] = numpy.zeros((1, 77, 768), dtype = numpy.float32)

	with thread_semaphore():
		noise_pred = unet.run(None, unet_inputs)[0][0]

	return noise_pred


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('diffusion_swapper_model')
	return create_static_model_set('full').get(model_name)


def process_frame(inputs : DiffusionSwapperInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frames = inputs.get('target_vision_frames')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	target_vision_frame = get_middle(target_vision_frames)
	target_faces = select_faces(reference_vision_frame, source_vision_frames, target_vision_frames)

	if target_faces:
		for target_face in target_faces:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			temp_vision_frame = swap_face(target_face, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
