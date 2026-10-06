from typing import List, Tuple

import numpy
import pytest

from facefusion import adaptive_masker, face_classifier, face_detector, face_landmarker, face_recognizer, state_manager, target_tracker
from facefusion.common_helper import get_middle
from facefusion.download import conditional_download
from facefusion.face_store import clear_faces
from facefusion.processors.modules.face_swapper import core as face_swapper
from facefusion.types import Embedding, Mask, VisionFrame
from facefusion.vision import read_static_images, select_video_frames
from .helper import get_test_example_file, get_test_examples_directory

TEST_FRAME_TOTAL = 12


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download(get_test_examples_directory(),
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-240p.mp4',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/source.jpg'
	])

	state_manager.init_item('execution_device_ids', [ 0 ])
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('download_providers', [ 'github' ])
	state_manager.init_item('reference_frame_number', 0)
	state_manager.init_item('face_detector_angles', [ 0 ])
	state_manager.init_item('face_detector_model', 'yolo_face')
	state_manager.init_item('face_detector_size', '640x640')
	state_manager.init_item('face_detector_margin', (0, 0, 0, 0))
	state_manager.init_item('face_detector_score', 0.5)
	state_manager.init_item('face_landmarker_model', '2dfan4')
	state_manager.init_item('face_landmarker_score', 0.5)
	state_manager.init_item('face_selector_mode', 'one')
	state_manager.init_item('face_selector_order', 'large-small')
	state_manager.init_item('reference_face_position', 0)
	state_manager.init_item('reference_face_distance', 0.3)
	state_manager.init_item('face_tracker_score', 0.0)
	state_manager.init_item('face_mask_types', [ 'box' ])
	state_manager.init_item('face_mask_areas', [])
	state_manager.init_item('face_mask_regions', [])
	state_manager.init_item('face_mask_blur', 0.3)
	state_manager.init_item('face_mask_padding', (0, 0, 0, 0))
	state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')
	state_manager.init_item('face_swapper_pixel_boost', '128x128')
	state_manager.init_item('face_swapper_weight', 0.5)
	state_manager.init_item('target_track', True)
	state_manager.init_item('target_track_max_lost_frames', 15)
	state_manager.init_item('target_track_smoothing', 0.5)
	state_manager.init_item('target_track_iou_threshold', 0.3)
	state_manager.init_item('target_track_embedding_threshold', 0.3)
	state_manager.init_item('target_track_motion_weight', 0.5)
	state_manager.init_item('target_track_identity_weight', 0.5)
	state_manager.init_item('target_track_debug', False)
	state_manager.init_item('adaptive_mask', True)
	state_manager.init_item('adaptive_mask_feather', 0.2)
	state_manager.init_item('temporal_mask', True)
	state_manager.init_item('partial_face_mode', True)

	face_classifier.pre_check()
	face_detector.pre_check()
	face_landmarker.pre_check()
	face_recognizer.pre_check()


@pytest.fixture(autouse = True)
def before_each(monkeypatch : pytest.MonkeyPatch) -> None:
	face_classifier.clear_inference_pool()
	face_detector.clear_inference_pool()
	face_landmarker.clear_inference_pool()
	face_recognizer.clear_inference_pool()
	clear_faces()
	target_tracker.clear_target_tracks()
	adaptive_masker.clear_adaptive_masks()
	state_manager.set_item('target_path', get_test_example_file('target-240p.mp4'))
	state_manager.set_item('target_track', True)
	monkeypatch.setattr(face_swapper, 'forward_swap_face', fake_forward_swap_face)


def fake_forward_swap_face(source_face : object, target_face : object, source_vision_frame : VisionFrame, crop_vision_frame : VisionFrame) -> VisionFrame:
	return numpy.zeros((3, crop_vision_frame.shape[2], crop_vision_frame.shape[3]), dtype = numpy.float32)


def process_target_frame(frame_number : int) -> Tuple[VisionFrame, Mask]:
	target_path = state_manager.get_item('target_path')
	source_vision_frames = read_static_images([ get_test_example_file('source.jpg') ])
	reference_vision_frame = select_video_frames(target_path, 0, 0)[0]
	target_vision_frames = select_video_frames(target_path, frame_number, 2)
	temp_vision_frame = get_middle(target_vision_frames)
	target_tracker.set_current_frame_number(frame_number)
	return face_swapper.process_frame(
	{
		'reference_vision_frame': reference_vision_frame,
		'source_vision_frames': source_vision_frames,
		'target_vision_frames': target_vision_frames,
		'temp_vision_frame': temp_vision_frame[:, :, :3],
		'temp_vision_mask': None
	})


def test_target_track_pipeline_keeps_identity_and_mask() -> None:
	statuses : List[str] = []

	for frame_number in range(TEST_FRAME_TOTAL):
		target_path = state_manager.get_item('target_path')
		temp_vision_frame = get_middle(select_video_frames(target_path, frame_number, 2))
		output_frame, _ = process_target_frame(frame_number)

		assert output_frame.shape == temp_vision_frame[:, :, :3].shape
		assert numpy.any(output_frame > 0)
		statuses.append(target_tracker.get_tracking_status())

	state = target_tracker.get_target_track_state()

	assert state is not None
	assert statuses.count('tracked') >= TEST_FRAME_TOTAL - 3
	assert 'predicted' not in statuses[:2]
	assert state.get('frame_index') >= TEST_FRAME_TOTAL - 3
	assert len(state.get('references')) >= 1


def test_target_track_pipeline_has_no_identity_switch() -> None:
	for frame_number in range(TEST_FRAME_TOTAL):
		process_target_frame(frame_number)

	state = target_tracker.get_target_track_state()
	references = state.get('references')
	embeddings : List[Embedding] = [ reference.get('embedding') for reference in references ]

	for index, embedding in enumerate(embeddings):
		for other_embedding in embeddings[index + 1:]:
			similarity = float(numpy.dot(embedding, other_embedding))
			assert similarity > 0.4, similarity


def test_target_track_pipeline_masks_are_stable() -> None:
	track_key = target_tracker.resolve_track_key()
	mask_masses : List[float] = []

	for frame_number in range(TEST_FRAME_TOTAL):
		process_target_frame(frame_number)
		stored_mask = adaptive_masker.ADAPTIVE_MASK_SET.get(track_key)

		assert stored_mask is not None
		assert float(stored_mask.max()) > 0
		mask_masses.append(float(stored_mask.sum()))

	for index, mask_mass in enumerate(mask_masses[1:]):
		ratio = mask_mass / mask_masses[index]
		assert 0.5 < ratio < 2.0, ( index, ratio )


def test_target_track_pipeline_disabled_matches_existing_behaviour(monkeypatch : pytest.MonkeyPatch) -> None:
	state_manager.set_item('target_track', False)
	state_manager.set_item('adaptive_mask', False)
	state_manager.set_item('temporal_mask', False)

	output_frame, _ = process_target_frame(0)
	temp_vision_frame = get_middle(select_video_frames(state_manager.get_item('target_path'), 0, 2))

	assert output_frame.shape == temp_vision_frame[:, :, :3].shape
	assert target_tracker.get_target_track_state() is None
