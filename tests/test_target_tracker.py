from typing import List

import numpy
import pytest

from facefusion import state_manager, target_tracker
from facefusion.face_store import clear_faces
from facefusion.types import Embedding, Face, FaceLandmarkSet


def create_test_embedding(seed : int, noise : float = 0.0) -> Embedding:
	rng = numpy.random.default_rng(seed)
	embedding = rng.standard_normal(512) + noise
	return embedding.astype(numpy.float64)


def normalize_embedding(embedding : Embedding) -> Embedding:
	return (embedding / numpy.linalg.norm(embedding)).astype(numpy.float64)


def create_test_face(center : List[float], size : float, embedding : Embedding) -> Face:
	x, y = center
	face_landmark_5 = numpy.array(
	[
		[ x - size * 0.25, y - size * 0.15 ],
		[ x + size * 0.25, y - size * 0.15 ],
		[ x, y ],
		[ x - size * 0.20, y + size * 0.30 ],
		[ x + size * 0.20, y + size * 0.30 ]
	], dtype = numpy.float64)
	angles = numpy.linspace(0, 2 * numpy.pi, 68, endpoint = False)
	face_landmark_68 = numpy.stack([ x + numpy.cos(angles) * size * 0.6, y + numpy.sin(angles) * size * 0.75 ], axis = 1)
	landmark_set : FaceLandmarkSet =\
	{
		'5': face_landmark_5,
		'5/68': face_landmark_5,
		'68': face_landmark_68,
		'68/5': face_landmark_68
	}
	bounding_box = numpy.array([ x - size * 0.6, y - size * 0.8, x + size * 0.6, y + size * 0.8 ], dtype = numpy.float64)
	return Face(
		origin = 'detect',
		bounding_box = bounding_box,
		score_set = { 'detector': 0.9, 'landmarker': 0.7 },
		landmark_set = landmark_set,
		angle = 0,
		embedding = embedding,
		embedding_norm = normalize_embedding(embedding),
		gender = 'female',
		age = range(20, 30),
		race = 'white'
	)


@pytest.fixture(autouse = True)
def before_each() -> None:
	clear_faces()
	target_tracker.clear_target_tracks()
	state_manager.init_item('target_path', 'test-target-path')
	state_manager.init_item('target_track', True)
	state_manager.init_item('target_track_max_lost_frames', 15)
	state_manager.init_item('target_track_smoothing', 0.5)
	state_manager.init_item('target_track_iou_threshold', 0.3)
	state_manager.init_item('target_track_embedding_threshold', 0.3)
	state_manager.init_item('target_track_motion_weight', 0.5)
	state_manager.init_item('target_track_identity_weight', 0.5)
	state_manager.init_item('target_track_debug', False)
	target_tracker.set_current_frame_number(0)


def test_lock_target_face_persists_identity() -> None:
	embedding = create_test_embedding(1)
	face = create_test_face([ 100, 100 ], 40, embedding)
	state = target_tracker.lock_target_face(face, 0)

	assert state.get('status') == 'tracked'
	assert state.get('frame_index') == 0
	assert state.get('last_bounding_box') is not None
	assert len(state.get('references')) == 1
	assert target_tracker.calculate_identity_score(face.embedding_norm, state) > 0.9
	assert target_tracker.calculate_identity_score(normalize_embedding(create_test_embedding(2)), state) < 0.3


def test_classify_pose() -> None:
	assert target_tracker.classify_pose(0, 0) == 'front'
	assert target_tracker.classify_pose(40, 0) == 'right'
	assert target_tracker.classify_pose(-40, 0) == 'left'
	assert target_tracker.classify_pose(0, 40) == 'up'
	assert target_tracker.classify_pose(0, -40) == 'down'


def test_register_reference_builds_multi_pose_identity() -> None:
	state = target_tracker.create_empty_state()
	target_tracker.register_reference(state, normalize_embedding(create_test_embedding(1)), 'front')
	target_tracker.register_reference(state, normalize_embedding(create_test_embedding(1, 0.2)), 'left')
	target_tracker.register_reference(state, normalize_embedding(create_test_embedding(1, 0.3)), 'right')

	assert len(state.get('references')) == 3
	assert state.get('target_identity') is not None


def test_select_target_face_keeps_tracked_face(monkeypatch : pytest.MonkeyPatch) -> None:
	embedding = create_test_embedding(1)
	face = create_test_face([ 100, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'bootstrap_target_face', lambda *args : face)
	reference_vision_frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	assert target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ]) == [ face ]

	moved_face = create_test_face([ 112, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'collect_window_faces', lambda *args : [ moved_face ])
	target_tracker.set_current_frame_number(1)
	tracked_faces = target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])

	assert len(tracked_faces) == 1
	assert target_tracker.get_tracking_status() == 'tracked'
	assert target_tracker.get_target_track_state().get('frame_index') == 1


def test_select_target_face_predicts_during_dropout(monkeypatch : pytest.MonkeyPatch) -> None:
	embedding = create_test_embedding(1)
	face = create_test_face([ 100, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'bootstrap_target_face', lambda *args : face)
	reference_vision_frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])
	monkeypatch.setattr(target_tracker, 'collect_window_faces', lambda *args : [])

	target_tracker.set_current_frame_number(1)
	predicted_faces = target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])

	assert len(predicted_faces) == 1
	assert predicted_faces[0].origin == 'track'
	assert target_tracker.get_tracking_status() == 'predicted'


def test_select_target_face_terminates_after_lost_frames(monkeypatch : pytest.MonkeyPatch) -> None:
	embedding = create_test_embedding(1)
	face = create_test_face([ 100, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'bootstrap_target_face', lambda *args : face)
	reference_vision_frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])
	monkeypatch.setattr(target_tracker, 'collect_window_faces', lambda *args : [])

	target_tracker.set_current_frame_number(40)
	lost_faces = target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])

	assert lost_faces == []
	assert target_tracker.get_tracking_status() == 'lost'


def test_select_target_face_rejects_other_identity(monkeypatch : pytest.MonkeyPatch) -> None:
	embedding = create_test_embedding(1)
	face = create_test_face([ 100, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'bootstrap_target_face', lambda *args : face)
	reference_vision_frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])
	other_face = create_test_face([ 400, 400 ], 40, create_test_embedding(9))
	monkeypatch.setattr(target_tracker, 'collect_window_faces', lambda *args : [ other_face ])

	target_tracker.set_current_frame_number(1)
	tracked_faces = target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ])

	assert len(tracked_faces) == 1
	assert target_tracker.get_tracking_status() == 'predicted'


def test_select_target_face_delegates_when_disabled(monkeypatch : pytest.MonkeyPatch) -> None:
	state_manager.set_item('target_track', False)
	embedding = create_test_embedding(1)
	sentinel_face = create_test_face([ 100, 100 ], 40, embedding)
	monkeypatch.setattr(target_tracker, 'select_faces', lambda *args : [ sentinel_face ])
	reference_vision_frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	assert target_tracker.select_target_face(reference_vision_frame, [], [ reference_vision_frame ]) == [ sentinel_face ]


def test_predict_observation_applies_motion() -> None:
	state = target_tracker.create_empty_state()
	moving_face = create_test_face([ 100, 100 ], 40, create_test_embedding(1))
	target_tracker.commit_observation(state, moving_face, 0, 1.0, 'tracked', 1.0)
	moved_face = create_test_face([ 110, 100 ], 40, create_test_embedding(1))
	target_tracker.commit_observation(state, moved_face, 1, 1.0, 'tracked', 1.0)

	predicted = target_tracker.predict_observation(state, 3)

	assert predicted is not None
	assert predicted.get('bounding_box')[0] > moved_face.bounding_box[0]


def test_draw_track_debug_marks_frame() -> None:
	state = target_tracker.create_empty_state()
	face = create_test_face([ 120, 120 ], 40, create_test_embedding(1))
	target_tracker.commit_observation(state, face, 0, 1.0, 'tracked', 1.0)
	frame = numpy.zeros((240, 240, 3), dtype = numpy.uint8)

	debug_frame = target_tracker.draw_track_debug(frame, face)

	assert debug_frame.shape == frame.shape
	assert numpy.any(debug_frame > 0)
