import threading
from typing import Dict

import cv2
import numpy

from facefusion import face_masker, state_manager, target_tracker
from facefusion.types import FaceLandmark5, Mask, Resolution, TrackingStatus, VisionFrame

ADAPTIVE_MASK_SET : Dict[str, Mask] = {}
ADAPTIVE_MASK_LOCK : threading.Lock = threading.Lock()
MAX_POSE_GAIN : float = 0.5


def clear_adaptive_masks() -> None:
	with ADAPTIVE_MASK_LOCK:
		ADAPTIVE_MASK_SET.clear()


def get_status_confidence(tracking_status : TrackingStatus) -> float:
	if tracking_status == 'tracked':
		return 1.0
	if tracking_status == 'reacquiring':
		return 0.7
	if tracking_status == 'predicted':
		return 0.35
	return 0.0


def create_posed_face_mask(crop_size : Resolution, face_landmark_5_crop : FaceLandmark5, yaw : float, pitch : float, roll : float) -> Mask:
	crop_width, crop_height = crop_size
	left_eye, right_eye, nose_tip, mouth_left, mouth_right = numpy.asarray(face_landmark_5_crop, dtype = numpy.float64)
	eyes_mid = (left_eye + right_eye) * 0.5
	mouth_mid = (mouth_left + mouth_right) * 0.5
	eye_distance = max(float(numpy.linalg.norm(right_eye - left_eye)), 1e-6)
	face_height = max(float(numpy.linalg.norm(mouth_mid - eyes_mid)) * 2.2, eye_distance)
	yaw_factor = float(numpy.sin(numpy.radians(yaw)))
	pitch_factor = float(numpy.sin(numpy.radians(pitch)))
	face_center = (left_eye + right_eye + nose_tip + mouth_left + mouth_right) / 5
	face_center = face_center + numpy.array([ yaw_factor * eye_distance * 0.30, -pitch_factor * face_height * 0.15 ])
	axis_x = eye_distance * (1.05 + 0.35 * abs(yaw_factor))
	axis_y = face_height * (0.62 + 0.18 * abs(pitch_factor))
	posed_mask : Mask = numpy.zeros((crop_height, crop_width), dtype = numpy.float32)
	cv2.ellipse(posed_mask, (int(round(face_center[0])), int(round(face_center[1]))), (int(round(axis_x)), int(round(axis_y))), float(roll), 0, 360, 1.0, -1) #type:ignore[call-overload]
	return posed_mask


def feather_mask(crop_mask : Mask, feather : float) -> Mask:
	if feather <= 0:
		return crop_mask

	blur_amount = int(min(crop_mask.shape[:2]) * 0.12 * feather)

	if blur_amount < 1:
		return crop_mask
	return cv2.GaussianBlur(crop_mask, (0, 0), blur_amount * 0.25).clip(0, 1)


def propagate_to_visible_border(crop_mask : Mask, face_landmark_5_crop : FaceLandmark5) -> Mask:
	crop_height, crop_width = crop_mask.shape[:2]
	row_indices, column_indices = numpy.where(crop_mask > 0.5)

	if column_indices.size == 0:
		return crop_mask

	result_mask = crop_mask.copy()
	margin = max(int(min(crop_width, crop_height) * 0.12), 1)
	column_profile = crop_mask.max(axis = 0)
	row_profile = crop_mask.max(axis = 1)
	x1, x2 = int(column_indices.min()), int(column_indices.max())
	y1, y2 = int(row_indices.min()), int(row_indices.max())

	if x1 <= margin:
		result_mask[:, :x1] = numpy.maximum(result_mask[:, :x1], column_profile[x1])
	if crop_width - x2 <= margin:
		result_mask[:, x2 + 1:] = numpy.maximum(result_mask[:, x2 + 1:], column_profile[x2])
	if y1 <= margin:
		result_mask[:y1, :] = numpy.maximum(result_mask[:y1, :], row_profile[y1])
	if crop_height - y2 <= margin:
		result_mask[y2 + 1:, :] = numpy.maximum(result_mask[y2 + 1:, :], row_profile[y2])
	return result_mask


def create_adaptive_crop_mask(crop_vision_frame : VisionFrame, crop_mask : Mask, face_landmark_5_crop : FaceLandmark5, tracking_status : TrackingStatus) -> Mask:
	adaptive_mask = state_manager.get_item('adaptive_mask')
	temporal_mask_view = state_manager.get_item('temporal_mask')

	if not adaptive_mask and not temporal_mask_view:
		return crop_mask.clip(0, 1)

	track_key = target_tracker.resolve_track_key()
	crop_size = crop_vision_frame.shape[:2][::-1]
	mask = crop_mask.astype(numpy.float32).clip(0, 1)
	yaw, pitch, roll = face_masker.estimate_head_pose(face_landmark_5_crop)

	if adaptive_mask:
		posed_mask = create_posed_face_mask(crop_size, face_landmark_5_crop, yaw, pitch, roll)
		pose_gain = min(numpy.interp(abs(yaw), [ 0, 75 ], [ 0.0, 0.35 ]) + numpy.interp(abs(pitch), [ 0, 60 ], [ 0.0, 0.25 ]), MAX_POSE_GAIN)
		support = cv2.dilate((mask > 0.5).astype(numpy.float32), numpy.ones((5, 5), numpy.uint8), iterations = 1)
		posed_union = numpy.maximum(mask, posed_mask * pose_gain)
		mask = numpy.where(support > 0.5, posed_union, mask)

	with ADAPTIVE_MASK_LOCK:
		previous_mask = ADAPTIVE_MASK_SET.get(track_key)

	if temporal_mask_view and previous_mask is not None and previous_mask.shape == mask.shape:
		keep = float(state_manager.get_item('target_track_smoothing')) * (1 - get_status_confidence(tracking_status))
		mask = mask * (1 - keep) + previous_mask * keep
		mask = numpy.maximum(mask, previous_mask * min(0.5, keep + 0.15))

		if float(mask.max()) <= 0 and float(previous_mask.max()) > 0:
			mask = previous_mask.copy()

	if state_manager.get_item('partial_face_mode') and tracking_status in [ 'predicted', 'reacquiring' ]:
		mask = propagate_to_visible_border(mask, face_landmark_5_crop)

	mask = feather_mask(mask, float(state_manager.get_item('adaptive_mask_feather')))
	mask = mask.clip(0, 1).astype(numpy.float32)

	with ADAPTIVE_MASK_LOCK:
		ADAPTIVE_MASK_SET[track_key] = mask
	return mask
