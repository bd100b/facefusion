from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step, calculate_int_step
from facefusion.processors.core import load_processor_module
from facefusion.processors.modules.diffusion_swapper import choices as diffusion_swapper_choices
from facefusion.uis.core import get_ui_component, register_ui_component

DIFFUSION_SWAPPER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
DIFFUSION_SWAPPER_STRENGTH_SLIDER : Optional[gradio.Slider] = None
DIFFUSION_SWAPPER_STEPS_SLIDER : Optional[gradio.Slider] = None
DIFFUSION_SWAPPER_SCALE_SLIDER : Optional[gradio.Slider] = None
DIFFUSION_SWAPPER_SEED_NUMBER : Optional[gradio.Number] = None


def render() -> None:
	global DIFFUSION_SWAPPER_MODEL_DROPDOWN
	global DIFFUSION_SWAPPER_STRENGTH_SLIDER
	global DIFFUSION_SWAPPER_STEPS_SLIDER
	global DIFFUSION_SWAPPER_SCALE_SLIDER
	global DIFFUSION_SWAPPER_SEED_NUMBER

	has_diffusion_swapper = 'diffusion_swapper' in state_manager.get_item('processors')
	DIFFUSION_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'facefusion.processors.modules.diffusion_swapper'),
		choices = diffusion_swapper_choices.diffusion_swapper_models,
		value = state_manager.get_item('diffusion_swapper_model'),
		visible = has_diffusion_swapper
	)
	DIFFUSION_SWAPPER_STRENGTH_SLIDER = gradio.Slider(
		label = translator.get('uis.strength_slider', 'facefusion.processors.modules.diffusion_swapper'),
		value = state_manager.get_item('diffusion_swapper_strength'),
		step = calculate_float_step(diffusion_swapper_choices.diffusion_swapper_strength_range),
		minimum = diffusion_swapper_choices.diffusion_swapper_strength_range[0],
		maximum = diffusion_swapper_choices.diffusion_swapper_strength_range[-1],
		visible = has_diffusion_swapper
	)
	DIFFUSION_SWAPPER_STEPS_SLIDER = gradio.Slider(
		label = translator.get('uis.steps_slider', 'facefusion.processors.modules.diffusion_swapper'),
		value = state_manager.get_item('diffusion_swapper_steps'),
		step = calculate_int_step(diffusion_swapper_choices.diffusion_swapper_steps_range),
		minimum = diffusion_swapper_choices.diffusion_swapper_steps_range[0],
		maximum = diffusion_swapper_choices.diffusion_swapper_steps_range[-1],
		visible = has_diffusion_swapper
	)
	DIFFUSION_SWAPPER_SCALE_SLIDER = gradio.Slider(
		label = translator.get('uis.scale_slider', 'facefusion.processors.modules.diffusion_swapper'),
		value = state_manager.get_item('diffusion_swapper_scale'),
		step = calculate_float_step(diffusion_swapper_choices.diffusion_swapper_scale_range),
		minimum = diffusion_swapper_choices.diffusion_swapper_scale_range[0],
		maximum = diffusion_swapper_choices.diffusion_swapper_scale_range[-1],
		visible = has_diffusion_swapper
	)
	DIFFUSION_SWAPPER_SEED_NUMBER = gradio.Number(
		label = translator.get('uis.seed_number', 'facefusion.processors.modules.diffusion_swapper'),
		value = state_manager.get_item('diffusion_swapper_seed'),
		precision = 0,
		visible = has_diffusion_swapper
	)
	register_ui_component('diffusion_swapper_model_dropdown', DIFFUSION_SWAPPER_MODEL_DROPDOWN)


def listen() -> None:
	DIFFUSION_SWAPPER_MODEL_DROPDOWN.change(update_diffusion_swapper_model, inputs = DIFFUSION_SWAPPER_MODEL_DROPDOWN, outputs = DIFFUSION_SWAPPER_MODEL_DROPDOWN)
	DIFFUSION_SWAPPER_STRENGTH_SLIDER.release(update_diffusion_swapper_strength, inputs = DIFFUSION_SWAPPER_STRENGTH_SLIDER)
	DIFFUSION_SWAPPER_STEPS_SLIDER.release(update_diffusion_swapper_steps, inputs = DIFFUSION_SWAPPER_STEPS_SLIDER)
	DIFFUSION_SWAPPER_SCALE_SLIDER.release(update_diffusion_swapper_scale, inputs = DIFFUSION_SWAPPER_SCALE_SLIDER)
	DIFFUSION_SWAPPER_SEED_NUMBER.change(update_diffusion_swapper_seed, inputs = DIFFUSION_SWAPPER_SEED_NUMBER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ DIFFUSION_SWAPPER_MODEL_DROPDOWN, DIFFUSION_SWAPPER_STRENGTH_SLIDER, DIFFUSION_SWAPPER_STEPS_SLIDER, DIFFUSION_SWAPPER_SCALE_SLIDER, DIFFUSION_SWAPPER_SEED_NUMBER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Number]:
	has_diffusion_swapper = 'diffusion_swapper' in processors
	return gradio.Dropdown(visible = has_diffusion_swapper), gradio.Slider(visible = has_diffusion_swapper), gradio.Slider(visible = has_diffusion_swapper), gradio.Slider(visible = has_diffusion_swapper), gradio.Number(visible = has_diffusion_swapper)


def update_diffusion_swapper_model(diffusion_swapper_model : str) -> gradio.Dropdown:
	diffusion_swapper_module = load_processor_module('diffusion_swapper')
	diffusion_swapper_module.clear_inference_pool()
	state_manager.set_item('diffusion_swapper_model', diffusion_swapper_model)

	if diffusion_swapper_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('diffusion_swapper_model'))
	return gradio.Dropdown()


def update_diffusion_swapper_strength(diffusion_swapper_strength : float) -> None:
	state_manager.set_item('diffusion_swapper_strength', diffusion_swapper_strength)


def update_diffusion_swapper_steps(diffusion_swapper_steps : int) -> None:
	state_manager.set_item('diffusion_swapper_steps', int(diffusion_swapper_steps))


def update_diffusion_swapper_scale(diffusion_swapper_scale : float) -> None:
	state_manager.set_item('diffusion_swapper_scale', diffusion_swapper_scale)


def update_diffusion_swapper_seed(diffusion_swapper_seed : int) -> None:
	state_manager.set_item('diffusion_swapper_seed', int(diffusion_swapper_seed))
