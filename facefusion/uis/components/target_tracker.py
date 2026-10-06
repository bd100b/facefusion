from typing import Optional

import gradio

import facefusion.choices
from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step, calculate_int_step
from facefusion.types import Score
from facefusion.uis.core import register_ui_component

TARGET_TRACK_CHECKBOX : Optional[gradio.Checkbox] = None
TARGET_TRACK_MAX_LOST_FRAMES_SLIDER : Optional[gradio.Slider] = None
TARGET_TRACK_SMOOTHING_SLIDER : Optional[gradio.Slider] = None
TARGET_TRACK_IDENTITY_WEIGHT_SLIDER : Optional[gradio.Slider] = None
ADAPTIVE_MASK_CHECKBOX : Optional[gradio.Checkbox] = None
ADAPTIVE_MASK_FEATHER_SLIDER : Optional[gradio.Slider] = None
TEMPORAL_MASK_CHECKBOX : Optional[gradio.Checkbox] = None
PARTIAL_FACE_MODE_CHECKBOX : Optional[gradio.Checkbox] = None


def render() -> None:
	global TARGET_TRACK_CHECKBOX
	global TARGET_TRACK_MAX_LOST_FRAMES_SLIDER
	global TARGET_TRACK_SMOOTHING_SLIDER
	global TARGET_TRACK_IDENTITY_WEIGHT_SLIDER
	global ADAPTIVE_MASK_CHECKBOX
	global ADAPTIVE_MASK_FEATHER_SLIDER
	global TEMPORAL_MASK_CHECKBOX
	global PARTIAL_FACE_MODE_CHECKBOX

	with gradio.Row():
		TARGET_TRACK_CHECKBOX = gradio.Checkbox(
			label = translator.get('uis.target_track_checkbox'),
			value = state_manager.get_item('target_track')
		)
		ADAPTIVE_MASK_CHECKBOX = gradio.Checkbox(
			label = translator.get('uis.adaptive_mask_checkbox'),
			value = state_manager.get_item('adaptive_mask')
		)
		TEMPORAL_MASK_CHECKBOX = gradio.Checkbox(
			label = translator.get('uis.temporal_mask_checkbox'),
			value = state_manager.get_item('temporal_mask')
		)
		PARTIAL_FACE_MODE_CHECKBOX = gradio.Checkbox(
			label = translator.get('uis.partial_face_mode_checkbox'),
			value = state_manager.get_item('partial_face_mode')
		)
	with gradio.Row():
		TARGET_TRACK_MAX_LOST_FRAMES_SLIDER = gradio.Slider(
			label = translator.get('uis.target_track_max_lost_frames_slider'),
			step = calculate_int_step(facefusion.choices.target_track_max_lost_frames_range),
			minimum = facefusion.choices.target_track_max_lost_frames_range[0],
			maximum = facefusion.choices.target_track_max_lost_frames_range[-1],
			value = state_manager.get_item('target_track_max_lost_frames')
		)
		TARGET_TRACK_SMOOTHING_SLIDER = gradio.Slider(
			label = translator.get('uis.target_track_smoothing_slider'),
			step = calculate_float_step(facefusion.choices.target_track_smoothing_range),
			minimum = facefusion.choices.target_track_smoothing_range[0],
			maximum = facefusion.choices.target_track_smoothing_range[-1],
			value = state_manager.get_item('target_track_smoothing')
		)
		TARGET_TRACK_IDENTITY_WEIGHT_SLIDER = gradio.Slider(
			label = translator.get('uis.target_track_identity_weight_slider'),
			step = calculate_float_step(facefusion.choices.target_track_identity_weight_range),
			minimum = facefusion.choices.target_track_identity_weight_range[0],
			maximum = facefusion.choices.target_track_identity_weight_range[-1],
			value = state_manager.get_item('target_track_identity_weight')
		)
		ADAPTIVE_MASK_FEATHER_SLIDER = gradio.Slider(
			label = translator.get('uis.adaptive_mask_feather_slider'),
			step = calculate_float_step(facefusion.choices.adaptive_mask_feather_range),
			minimum = facefusion.choices.adaptive_mask_feather_range[0],
			maximum = facefusion.choices.adaptive_mask_feather_range[-1],
			value = state_manager.get_item('adaptive_mask_feather')
		)
	register_ui_component('target_track_checkbox', TARGET_TRACK_CHECKBOX)
	register_ui_component('target_track_max_lost_frames_slider', TARGET_TRACK_MAX_LOST_FRAMES_SLIDER)
	register_ui_component('target_track_smoothing_slider', TARGET_TRACK_SMOOTHING_SLIDER)
	register_ui_component('target_track_identity_weight_slider', TARGET_TRACK_IDENTITY_WEIGHT_SLIDER)
	register_ui_component('adaptive_mask_checkbox', ADAPTIVE_MASK_CHECKBOX)
	register_ui_component('adaptive_mask_feather_slider', ADAPTIVE_MASK_FEATHER_SLIDER)
	register_ui_component('temporal_mask_checkbox', TEMPORAL_MASK_CHECKBOX)
	register_ui_component('partial_face_mode_checkbox', PARTIAL_FACE_MODE_CHECKBOX)


def listen() -> None:
	TARGET_TRACK_CHECKBOX.change(update_target_track, inputs = TARGET_TRACK_CHECKBOX)
	TARGET_TRACK_MAX_LOST_FRAMES_SLIDER.release(update_target_track_max_lost_frames, inputs = TARGET_TRACK_MAX_LOST_FRAMES_SLIDER)
	TARGET_TRACK_SMOOTHING_SLIDER.release(update_target_track_smoothing, inputs = TARGET_TRACK_SMOOTHING_SLIDER)
	TARGET_TRACK_IDENTITY_WEIGHT_SLIDER.release(update_target_track_identity_weight, inputs = TARGET_TRACK_IDENTITY_WEIGHT_SLIDER)
	ADAPTIVE_MASK_CHECKBOX.change(update_adaptive_mask, inputs = ADAPTIVE_MASK_CHECKBOX)
	ADAPTIVE_MASK_FEATHER_SLIDER.release(update_adaptive_mask_feather, inputs = ADAPTIVE_MASK_FEATHER_SLIDER)
	TEMPORAL_MASK_CHECKBOX.change(update_temporal_mask, inputs = TEMPORAL_MASK_CHECKBOX)
	PARTIAL_FACE_MODE_CHECKBOX.change(update_partial_face_mode, inputs = PARTIAL_FACE_MODE_CHECKBOX)


def update_target_track(target_track : bool) -> None:
	state_manager.set_item('target_track', target_track)


def update_target_track_max_lost_frames(target_track_max_lost_frames : int) -> None:
	state_manager.set_item('target_track_max_lost_frames', int(target_track_max_lost_frames))


def update_target_track_smoothing(target_track_smoothing : Score) -> None:
	state_manager.set_item('target_track_smoothing', target_track_smoothing)


def update_target_track_identity_weight(target_track_identity_weight : Score) -> None:
	state_manager.set_item('target_track_identity_weight', target_track_identity_weight)


def update_adaptive_mask(adaptive_mask : bool) -> None:
	state_manager.set_item('adaptive_mask', adaptive_mask)


def update_adaptive_mask_feather(adaptive_mask_feather : Score) -> None:
	state_manager.set_item('adaptive_mask_feather', adaptive_mask_feather)


def update_temporal_mask(temporal_mask : bool) -> None:
	state_manager.set_item('temporal_mask', temporal_mask)


def update_partial_face_mode(partial_face_mode : bool) -> None:
	state_manager.set_item('partial_face_mode', partial_face_mode)
