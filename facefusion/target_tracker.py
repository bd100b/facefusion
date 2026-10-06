import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

from facefusion import face_masker, state_manager
from facefusion.common_helper import get_middle
from facefusion.face_creator import get_one_face, get_static_faces
from facefusion.face_helper import calculate_bounding_box_overlap, estimate_face_angle, normalize_bounding_box
from facefusion.face_selector import select_faces, sort_faces_by_order
from facefusion.types import BoundingBox, Embedding, Face, FaceLandmarkSet, Points, PoseBucket, TargetTrackObservation, TargetTrackState, TargetTrackStateSet, TrackingStatus, VisionFrame

DEFAULT_TRACK_KEY : str = 'default'
MAX_OBSERVATIONS : int = 24
MAX_REFERENCES : int = 16
MAX_REFERENCES_PER_POSE : int = 4

TRACK_STATE_SET : TargetTrackStateSet = {}
TRACK_FACE_SET : Dict[str, Face] = {}
TRACK_STATE_LOCK : threading.Lock = threading.Lock()
CURRENT_FRAME_SET : threading.local = threading.local()


def clear_target_tracks() -> None:
	with TRACK_STATE_LOCK:
		TRACK_STATE_SET.clear()
		TRACK_FACE_SET.clear()


def resolve_track_key() -> str:
	target_path = state_manager.get_item('target_path')

	if isinstance(target_path, str):
		return target_path
	return DEFAULT_TRACK_KEY


def set_current_frame_number(frame_number : int) -> None:
	setattr(CURRENT_FRAME_SET, 'frame_number', frame_number)


def get_current_frame_number() -> int:
	return getattr(CURRENT_FRAME_SET, 'frame_number', 0)


def get_target_track_state() -> Optional[TargetTrackState]:
	with TRACK_STATE_LOCK:
		return TRACK_STATE_SET.get(resolve_track_key())


def get_tracking_status() -> TrackingStatus:
	state = get_target_track_state()

	if state:
		return state.get('status')
	return 'tracked'


def create_empty_state() -> TargetTrackState:
	return\
	{
		'target_identity': numpy.zeros((512,)).astype(numpy.float64),
		'references': [],
		'observations': [],
		'last_bounding_box': None,
		'last_landmark_set': None,
		'last_embedding': None,
		'last_affine_matrix': None,
		'last_mask': None,
		'face_confidence': 0.0,
		'frame_index': -1,
		'tracking_confidence': 0.0,
		'predicted_bounding_box': None,
		'estimated_scale': 0.0,
		'estimated_rotation': 0.0,
		'estimated_yaw': 0.0,
		'estimated_pitch': 0.0,
		'visibility': 0.0,
		'status': 'lost',
		'lost_frame_count': 0
	}


def cosine_similarity(embedding_a : Embedding, embedding_b : Embedding) -> float:
	if embedding_a is None or embedding_b is None:
		return 0.0

	norm_a = numpy.linalg.norm(embedding_a)
	norm_b = numpy.linalg.norm(embedding_b)

	if norm_a < 1e-9 or norm_b < 1e-9:
		return 0.0
	return float(numpy.clip(numpy.dot(embedding_a, embedding_b) / (norm_a * norm_b), -1.0, 1.0))


def calculate_identity_score(embedding_norm : Optional[Embedding], state : TargetTrackState) -> float:
	if embedding_norm is None:
		return 0.0

	best_similarity = 0.0

	for reference in state.get('references'):
		similarity = cosine_similarity(embedding_norm, reference.get('embedding'))
		best_similarity = max(best_similarity, similarity)

	target_similarity = cosine_similarity(embedding_norm, state.get('target_identity'))
	best_similarity = max(best_similarity, target_similarity)
	return float(numpy.clip(best_similarity, 0.0, 1.0))


def classify_pose(yaw : float, pitch : float) -> PoseBucket:
	if pitch > 25:
		return 'up'
	if pitch < -25:
		return 'down'
	if yaw > 20:
		return 'right'
	if yaw < -20:
		return 'left'
	return 'front'


def register_reference(state : TargetTrackState, embedding_norm : Embedding, pose : PoseBucket) -> None:
	references = state.get('references')
	total_per_pose = 0

	for index, reference in enumerate(references):
		if reference.get('pose') == pose:
			total_per_pose += 1

			if total_per_pose > MAX_REFERENCES_PER_POSE:
				references.pop(index)
				break

	references.append({ 'pose': pose, 'embedding': embedding_norm })

	while len(references) > MAX_REFERENCES:
		references.pop(0)

	target_embedding = numpy.mean([ reference.get('embedding') for reference in references ], axis = 0)
	state['target_identity'] = target_embedding.astype(numpy.float64)


def calculate_scale_consistency(bounding_box : BoundingBox, reference_bounding_box : BoundingBox) -> float:
	area = max((bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]), 1e-6)
	reference_area = max((reference_bounding_box[2] - reference_bounding_box[0]) * (reference_bounding_box[3] - reference_bounding_box[1]), 1e-6)
	return float(min(area, reference_area) / max(area, reference_area))


def calculate_motion_consistency(bounding_box : BoundingBox, reference_bounding_box : BoundingBox, velocity : Optional[BoundingBox]) -> float:
	if velocity is None:
		return 1.0

	center = (bounding_box[:2] + bounding_box[2:]) * 0.5
	reference_center = (reference_bounding_box[:2] + reference_bounding_box[2:]) * 0.5
	reference_velocity = (velocity[:2] + velocity[2:]) * 0.5
	moved = center - reference_center
	moved_norm = numpy.linalg.norm(moved)
	velocity_norm = numpy.linalg.norm(reference_velocity)

	if moved_norm < 1e-6 or velocity_norm < 1e-6:
		return 1.0
	return float(numpy.clip((numpy.dot(moved, reference_velocity) / (moved_norm * velocity_norm) + 1) * 0.5, 0.0, 1.0))


def estimate_velocity(state : TargetTrackState, frame_number : int) -> Optional[BoundingBox]:
	observations = [ observation for observation in state.get('observations') if observation.get('frame_number') < frame_number ]

	if len(observations) < 2:
		return None

	last_observation = observations[-1]
	previous_observation = observations[-2]
	frame_gap = last_observation.get('frame_number') - previous_observation.get('frame_number')

	if frame_gap <= 0:
		return None

	motion_weight = float(state_manager.get_item('target_track_motion_weight'))
	velocity = (last_observation.get('bounding_box') - previous_observation.get('bounding_box')) / frame_gap
	return (velocity * motion_weight).astype(numpy.float64)


def translate_landmark_set(landmark_set : FaceLandmarkSet, translation : Points) -> FaceLandmarkSet:
	return\
	{
		'5': landmark_set.get('5') + translation,
		'5/68': landmark_set.get('5/68') + translation,
		'68': landmark_set.get('68') + translation,
		'68/5': landmark_set.get('68/5') + translation
	}


def predict_observation(state : TargetTrackState, frame_number : int) -> Optional[TargetTrackObservation]:
	observations = [ observation for observation in state.get('observations') if observation.get('frame_number') < frame_number ]

	if not observations:
		return None

	last_observation = observations[-1]
	velocity = estimate_velocity(state, frame_number)
	frame_gap = max(frame_number - last_observation.get('frame_number'), 0)
	bounding_box = last_observation.get('bounding_box').copy()

	if velocity is not None:
		bounding_box = bounding_box + velocity * frame_gap

	bounding_box = normalize_bounding_box(bounding_box)
	translation = (bounding_box[:2] + bounding_box[2:]) * 0.5 - (last_observation.get('bounding_box')[:2] + last_observation.get('bounding_box')[2:]) * 0.5
	landmark_set = translate_landmark_set(last_observation.get('landmark_set'), translation)
	return\
	{
		'frame_number': frame_number,
		'bounding_box': bounding_box,
		'landmark_set': landmark_set,
		'embedding': last_observation.get('embedding'),
		'detector_score': last_observation.get('detector_score'),
		'identity_score': 1.0,
		'visibility': last_observation.get('visibility')
	}


def observation_from_state(state : TargetTrackState, frame_number : int) -> Optional[TargetTrackObservation]:
	last_bounding_box = state.get('last_bounding_box')
	last_landmark_set = state.get('last_landmark_set')
	last_embedding = state.get('last_embedding')

	if last_bounding_box is None or last_landmark_set is None or last_embedding is None:
		return None
	return\
	{
		'frame_number': frame_number,
		'bounding_box': last_bounding_box,
		'landmark_set': last_landmark_set,
		'embedding': last_embedding,
		'detector_score': state.get('face_confidence'),
		'identity_score': 1.0,
		'visibility': state.get('visibility')
	}


def calculate_candidate_score(face : Face, state : TargetTrackState, predicted : TargetTrackObservation, velocity : Optional[BoundingBox]) -> Tuple[float, float, float]:
	identity_score = calculate_identity_score(face.embedding_norm, state)
	spatial_score = calculate_bounding_box_overlap(face.bounding_box, predicted.get('bounding_box'))
	scale_score = calculate_scale_consistency(face.bounding_box, predicted.get('bounding_box'))
	motion_score = calculate_motion_consistency(face.bounding_box, predicted.get('bounding_box'), velocity)
	identity_weight = float(state_manager.get_item('target_track_identity_weight'))
	geometric_score = 0.6 * spatial_score + 0.25 * scale_score + 0.15 * motion_score
	combined_score = identity_weight * identity_score + (1 - identity_weight) * geometric_score
	return identity_score, spatial_score, combined_score


def create_predicted_face(state : TargetTrackState, observation : TargetTrackObservation) -> Optional[Face]:
	with TRACK_STATE_LOCK:
		last_face = TRACK_FACE_SET.get(resolve_track_key())

	if last_face is None:
		return None
	return last_face._replace(
		origin = 'track',
		bounding_box = observation.get('bounding_box'),
		landmark_set = observation.get('landmark_set'),
		angle = estimate_face_angle(observation.get('landmark_set').get('68/5'))
	)


def smooth_bounding_box(observed_bounding_box : BoundingBox, predicted_bounding_box : BoundingBox) -> BoundingBox:
	smoothing = float(state_manager.get_item('target_track_smoothing'))
	observed_center = (observed_bounding_box[:2] + observed_bounding_box[2:]) * 0.5
	predicted_center = (predicted_bounding_box[:2] + predicted_bounding_box[2:]) * 0.5
	diagonal = max(numpy.linalg.norm(predicted_bounding_box[2:] - predicted_bounding_box[:2]), 1e-6)
	motion_magnitude = numpy.linalg.norm(observed_center - predicted_center) / diagonal
	observed_weight = 1 - smoothing * 0.8

	if motion_magnitude > 0.15:
		observed_weight = max(observed_weight, 0.85)

	double_smoothed_bounding_box = predicted_bounding_box * (1 - observed_weight) + observed_bounding_box * observed_weight
	return normalize_bounding_box(double_smoothed_bounding_box)


def smooth_landmark_set(observed_landmark_set : FaceLandmarkSet, predicted_landmark_set : FaceLandmarkSet) -> FaceLandmarkSet:
	smoothing = float(state_manager.get_item('target_track_smoothing'))
	observed_weight = 1 - smoothing * 0.8
	return\
	{
		'5': predicted_landmark_set.get('5') * (1 - observed_weight) + observed_landmark_set.get('5') * observed_weight,
		'5/68': predicted_landmark_set.get('5/68') * (1 - observed_weight) + observed_landmark_set.get('5/68') * observed_weight,
		'68': predicted_landmark_set.get('68') * (1 - observed_weight) + observed_landmark_set.get('68') * observed_weight,
		'68/5': predicted_landmark_set.get('68/5') * (1 - observed_weight) + observed_landmark_set.get('68/5') * observed_weight
	}


def commit_observation(state : TargetTrackState, face : Face, frame_number : int, identity_score : float, status : TrackingStatus, visibility : float) -> None:
	yaw, pitch, roll = face_masker.estimate_head_pose(face.landmark_set.get('5'))
	pose = classify_pose(yaw, pitch)
	register_reference(state, face.embedding_norm, pose)
	observation : TargetTrackObservation =\
	{
		'frame_number': frame_number,
		'bounding_box': face.bounding_box,
		'landmark_set': face.landmark_set,
		'embedding': face.embedding_norm,
		'detector_score': face.score_set.get('detector'),
		'identity_score': identity_score,
		'visibility': visibility
	}
	observations = [ observation_item for observation_item in state.get('observations') if observation_item.get('frame_number') != frame_number ]
	observations.append(observation)
	observations = sorted(observations, key = lambda observation_item: observation_item.get('frame_number'))[-MAX_OBSERVATIONS:]
	state['observations'] = observations
	state['last_bounding_box'] = face.bounding_box
	state['last_landmark_set'] = face.landmark_set
	state['last_embedding'] = face.embedding
	state['face_confidence'] = float(face.score_set.get('detector'))
	state['frame_index'] = max(state.get('frame_index'), frame_number)
	state['tracking_confidence'] = float(numpy.clip(identity_score, 0.0, 1.0))
	state['predicted_bounding_box'] = face.bounding_box
	state['estimated_scale'] = float(max(face.bounding_box[2] - face.bounding_box[0], face.bounding_box[3] - face.bounding_box[1]))
	state['estimated_rotation'] = float(roll)
	state['estimated_yaw'] = float(yaw)
	state['estimated_pitch'] = float(pitch)
	state['visibility'] = visibility
	state['status'] = status
	state['lost_frame_count'] = 0

	with TRACK_STATE_LOCK:
		TRACK_FACE_SET[resolve_track_key()] = face
		TRACK_STATE_SET[resolve_track_key()] = state


def lock_target_face(face : Face, frame_number : int) -> TargetTrackState:
	state = create_empty_state()
	commit_observation(state, face, frame_number, 1.0, 'tracked', 1.0)
	return state


def collect_window_faces(target_vision_frames : List[VisionFrame]) -> List[Face]:
	window_faces : List[Face] = []

	for target_vision_frame in target_vision_frames:
		if target_vision_frame is not None:
			window_faces.extend(get_static_faces([ target_vision_frame ]))
	return window_faces


def select_best_candidate(candidates : List[Face], state : TargetTrackState, frame_number : int) -> Tuple[Optional[Face], float, float, float]:
	predicted = predict_observation(state, frame_number)

	if predicted is None:
		predicted = observation_from_state(state, frame_number)

	if predicted is None:
		return candidates[0] if candidates else None, 0.0, 0.0, 0.0

	velocity = estimate_velocity(state, frame_number)
	embedding_threshold = float(state_manager.get_item('target_track_embedding_threshold'))
	iou_threshold = float(state_manager.get_item('target_track_iou_threshold'))
	best_face : Optional[Face] = None
	best_identity = 0.0
	best_spatial = 0.0
	best_combined = 0.0

	for candidate in candidates:
		identity_score, spatial_score, combined_score = calculate_candidate_score(candidate, state, predicted, velocity)

		if identity_score >= embedding_threshold or spatial_score >= iou_threshold:
			if combined_score > best_combined:
				best_face = candidate
				best_identity = identity_score
				best_spatial = spatial_score
				best_combined = combined_score

	return best_face, best_identity, best_spatial, best_combined


def bootstrap_target_face(reference_vision_frame : VisionFrame, source_vision_frames : List[VisionFrame], target_vision_frames : List[VisionFrame]) -> Optional[Face]:
	target_face = get_one_face(select_faces(reference_vision_frame, source_vision_frames, target_vision_frames))

	if target_face is None:
		target_vision_frame = get_middle(target_vision_frames)

		if target_vision_frame is not None:
			target_face = get_one_face(sort_faces_by_order(get_static_faces([ target_vision_frame ]), 'large-small'))
	return target_face


def select_target_face(reference_vision_frame : VisionFrame, source_vision_frames : List[VisionFrame], target_vision_frames : List[VisionFrame]) -> List[Face]:
	if not state_manager.get_item('target_track'):
		return select_faces(reference_vision_frame, source_vision_frames, target_vision_frames)

	track_key = resolve_track_key()
	frame_number = get_current_frame_number()

	with TRACK_STATE_LOCK:
		state = TRACK_STATE_SET.get(track_key)

	if state is None:
		target_face = bootstrap_target_face(reference_vision_frame, source_vision_frames, target_vision_frames)

		if target_face is None:
			return []

		lock_target_face(target_face, frame_number)
		return [ target_face ]

	candidates = collect_window_faces(target_vision_frames)
	best_face, identity_score, spatial_score, combined_score = select_best_candidate(candidates, state, frame_number)
	embedding_threshold = float(state_manager.get_item('target_track_embedding_threshold'))
	iou_threshold = float(state_manager.get_item('target_track_iou_threshold'))

	if best_face is not None:
		predicted = predict_observation(state, frame_number)

		if predicted is None:
			predicted = observation_from_state(state, frame_number)

		if predicted is not None:
			observed_bounding_box = smooth_bounding_box(best_face.bounding_box, predicted.get('bounding_box'))
			observed_landmark_set = smooth_landmark_set(best_face.landmark_set, predicted.get('landmark_set'))
			best_face = best_face._replace(bounding_box = observed_bounding_box, landmark_set = observed_landmark_set)

		if identity_score >= embedding_threshold and spatial_score >= iou_threshold:
			status : TrackingStatus = 'tracked'
		else:
			status = 'reacquiring'

		visibility = float(numpy.clip(0.6 * spatial_score + 0.4 * identity_score, 0.0, 1.0))
		commit_observation(state, best_face, frame_number, identity_score, status, visibility)
		return [ best_face ]

	lost_frame_count = max(frame_number - state.get('frame_index'), 0)
	max_lost_frames = int(state_manager.get_item('target_track_max_lost_frames'))

	if state.get('frame_index') < 0 or lost_frame_count > max_lost_frames:
		state['status'] = 'lost'
		state['lost_frame_count'] = lost_frame_count
		state['tracking_confidence'] = 0.0

		with TRACK_STATE_LOCK:
			TRACK_STATE_SET[track_key] = state
		return []

	predicted = predict_observation(state, frame_number)

	if predicted is None:
		predicted = observation_from_state(state, frame_number)

	if predicted is None:
		return []

	predicted_face = create_predicted_face(state, predicted)
	state['status'] = 'predicted'
	state['lost_frame_count'] = lost_frame_count
	state['tracking_confidence'] = float(numpy.clip(1 - lost_frame_count / max(max_lost_frames, 1), 0.0, 1.0))

	with TRACK_STATE_LOCK:
		TRACK_STATE_SET[track_key] = state

	if predicted_face is None:
		return []
	return [ predicted_face ]


def calculate_debug_scale(temp_vision_frame : VisionFrame) -> int:
	frame_height = temp_vision_frame.shape[0]
	return max(1, min(10, round(frame_height / 270)))


def draw_track_debug(temp_vision_frame : VisionFrame, observed_face : Optional[Face] = None) -> VisionFrame:
	state = get_target_track_state()

	if state is None:
		return temp_vision_frame

	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	border_scale = calculate_debug_scale(temp_vision_frame)
	box_color = 0, 0, 255
	track_color = 0, 255, 0
	predict_color = 255, 160, 0
	text_color = 255, 255, 255

	if observed_face is not None:
		observed_box = observed_face.bounding_box.astype(numpy.int32)
		cv2.rectangle(temp_vision_frame, (observed_box[0], observed_box[1]), (observed_box[2], observed_box[3]), box_color, border_scale)

	if state.get('last_bounding_box') is not None:
		track_box = state.get('last_bounding_box').astype(numpy.int32)
		cv2.rectangle(temp_vision_frame, (track_box[0], track_box[1]), (track_box[2], track_box[3]), track_color, border_scale)

	if state.get('predicted_bounding_box') is not None and state.get('status') in [ 'predicted', 'reacquiring' ]:
		predict_box = state.get('predicted_bounding_box').astype(numpy.int32)
		cv2.rectangle(temp_vision_frame, (predict_box[0], predict_box[1]), (predict_box[2], predict_box[3]), predict_color, border_scale)

	if state.get('last_landmark_set') is not None:
		for point in state.get('last_landmark_set').get('5').astype(numpy.int32):
			cv2.circle(temp_vision_frame, (point[0], point[1]), border_scale, track_color, -1)

	status_lines =\
	[
		'state: ' + state.get('status'),
		'track: ' + str(round(state.get('tracking_confidence'), 2)),
		'detector: ' + str(round(state.get('face_confidence'), 2)),
		'pose: ' + str(round(state.get('estimated_yaw'))) + '/' + str(round(state.get('estimated_pitch'))) + '/' + str(round(state.get('estimated_rotation'))),
		'lost: ' + str(state.get('lost_frame_count'))
	]
	line_height = (border_scale + 1) * 14
	text_position_y = (border_scale + 1) * 20

	for status_line in status_lines:
		cv2.putText(temp_vision_frame, status_line, (text_position_y, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * border_scale, text_color, max(border_scale - 1, 1))
		text_position_y += line_height

	return temp_vision_frame
