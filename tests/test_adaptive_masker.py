import numpy
import pytest

from facefusion import adaptive_masker, state_manager
from facefusion.face_store import clear_faces
from facefusion.types import FaceLandmark5, VisionFrame


def create_test_landmark_5(center : float) -> FaceLandmark5:
	return numpy.array(
	[
		[ center - 20, center - 12 ],
		[ center + 20, center - 12 ],
		[ center, center ],
		[ center - 16, center + 24 ],
		[ center + 16, center + 24 ]
	], dtype = numpy.float64)


@pytest.fixture(autouse = True)
def before_each() -> None:
	clear_faces()
	adaptive_masker.clear_adaptive_masks()
	state_manager.init_item('target_path', 'test-mask-path')
	state_manager.init_item('adaptive_mask', True)
	state_manager.init_item('adaptive_mask_feather', 0.2)
	state_manager.init_item('temporal_mask', True)
	state_manager.init_item('partial_face_mode', True)
	state_manager.init_item('target_track_smoothing', 0.5)


def create_test_crop() -> VisionFrame:
	return numpy.zeros((256, 256, 3), dtype = numpy.uint8)


def test_create_posed_face_mask_covers_center() -> None:
	posed_mask = adaptive_masker.create_posed_face_mask((256, 256), create_test_landmark_5(128), 0, 0, 0)

	assert posed_mask.shape == (256, 256)
	assert posed_mask[128, 128] == 1.0
	assert posed_mask.max() == 1.0


def test_create_posed_face_mask_adapts_to_yaw() -> None:
	front_mask = adaptive_masker.create_posed_face_mask((256, 256), create_test_landmark_5(128), 0, 0, 0)
	yaw_mask = adaptive_masker.create_posed_face_mask((256, 256), create_test_landmark_5(128), 60, 0, 0)

	assert yaw_mask.sum() > front_mask.sum()


def test_feather_mask_softens_edges() -> None:
	crop_mask = numpy.zeros((256, 256), dtype = numpy.float32)
	crop_mask[80:176, 80:176] = 1.0

	feathered_mask = adaptive_masker.feather_mask(crop_mask, 0.8)

	assert feathered_mask.shape == crop_mask.shape
	assert feathered_mask.max() <= 1.0
	assert feathered_mask.min() < 0.0 or numpy.any((feathered_mask > 0) & (feathered_mask < crop_mask))


def test_propagate_to_visible_border_extends_mask() -> None:
	crop_mask = numpy.zeros((256, 256), dtype = numpy.float32)
	crop_mask[100:150, 4:60] = 1.0

	propagated_mask = adaptive_masker.propagate_to_visible_border(crop_mask, create_test_landmark_5(32))

	assert propagated_mask[:, 0].max() > 0


def test_create_adaptive_crop_mask_returns_input_when_disabled() -> None:
	state_manager.set_item('adaptive_mask', False)
	state_manager.set_item('temporal_mask', False)
	crop_vision_frame = create_test_crop()
	crop_mask = numpy.ones((256, 256), dtype = numpy.float32)

	result_mask = adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, crop_mask, create_test_landmark_5(128), 'tracked')

	assert numpy.array_equal(result_mask, crop_mask.clip(0, 1))


def test_temporal_mask_propagates_previous_frame() -> None:
	crop_vision_frame = create_test_crop()
	full_mask = numpy.zeros((256, 256), dtype = numpy.float32)
	full_mask[80:176, 80:176] = 1.0

	first_mask = adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, full_mask, create_test_landmark_5(128), 'tracked')
	empty_mask = numpy.zeros((256, 256), dtype = numpy.float32)
	second_mask = adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, empty_mask, create_test_landmark_5(128), 'predicted')

	assert first_mask.max() > 0
	assert second_mask.max() > 0
	assert second_mask.sum() > 0


def test_create_adaptive_crop_mask_restores_previous_when_tracked() -> None:
	crop_vision_frame = create_test_crop()
	crop_mask = numpy.zeros((256, 256), dtype = numpy.float32)
	crop_mask[80:176, 80:176] = 1.0

	adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, crop_mask, create_test_landmark_5(128), 'tracked')
	result_mask = adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, numpy.zeros((256, 256), dtype = numpy.float32), create_test_landmark_5(128), 'tracked')

	assert result_mask.shape == (256, 256)
	assert result_mask.sum() > 0
