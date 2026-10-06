"""Reproducible demo for the persistent target tracker.

It runs the real FaceFusion detector, landmarker and recognizer over a real
video and compares:

  OLD : the stock per frame face selector (face_selector.select_faces)
  NEW : the persistent target tracker (target_tracker.select_target_face)

The face swapper model used by the identity swap (inswapper_128_fp16) does not
fit into the memory of this demo machine, so the swapped pixels themselves are
not rendered. Instead the demo renders the two things the tracker is
responsible for: which face is selected as the target, and which region the
adaptive mask covers. It also injects two temporary detection dropouts so the
lost frame tolerance is visible.

Usage
-----
python demo/target_track_demo.py            # render, assemble and report
python demo/target_track_demo.py --start 0 --end 90   # render one chunk only
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facefusion import adaptive_masker, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager, target_tracker  # noqa: E402
from facefusion.face_creator import get_static_faces  # noqa: E402
from facefusion.face_selector import select_faces  # noqa: E402
from facefusion.types import Face, TrackingStatus, VisionFrame  # noqa: E402
from facefusion.vision import read_static_images, read_static_video_frame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.environ.get('DEMO_TARGET', '/tmp/facefusion-test-examples/target-240p.mp4')
SOURCE_PATH = os.environ.get('DEMO_SOURCE', '/tmp/facefusion-test-examples/source.jpg')
FRAME_TOTAL = 270
FRAME_RATE = 25.0

# two simulated detection dropouts: the first stays inside the lost frame
# budget, the second deliberately exceeds it
DROPOUTS = [ (88, 100), (180, 202) ]
MAX_LOST_FRAMES = 15

PANEL_WIDTH = 480
PANEL_HEIGHT = 255
HEADER_HEIGHT = 36
FOOTER_HEIGHT = 55
TICK_WIDTH = 3

OLD_TITLE = 'OLD: per frame selector'
NEW_TITLE = 'NEW: persistent target tracker'
STATUS_COLOR : Dict[str, tuple] =\
{
	'tracked': (90, 230, 90),
	'reacquiring': (60, 210, 230),
	'predicted': (40, 170, 255),
	'lost': (70, 70, 240)
}


def init_state() -> None:
	state_item : Dict[str, object] =\
	{
		'execution_device_ids': [ 0 ],
		'execution_providers': [ 'cpu' ],
		'download_providers': [ 'github' ],
		'reference_frame_number': 0,
		'face_detector_angles': [ 0 ],
		'face_detector_model': 'yolo_face',
		'face_detector_size': '640x640',
		'face_detector_margin': (0, 0, 0, 0),
		'face_detector_score': 0.5,
		'face_landmarker_model': '2dfan4',
		'face_landmarker_score': 0.5,
		'face_selector_mode': 'one',
		'face_selector_order': 'large-small',
		'reference_face_position': 0,
		'reference_face_distance': 0.3,
		'face_tracker_score': 0.0,
		'face_mask_types': [ 'box' ],
		'face_mask_areas': [],
		'face_mask_regions': [],
		'face_mask_blur': 0.3,
		'face_mask_padding': (0, 0, 0, 0),
		'face_swapper_model': 'inswapper_128_fp16',
		'face_swapper_pixel_boost': '128x128',
		'face_swapper_weight': 0.5,
		'target_track': True,
		'target_track_max_lost_frames': MAX_LOST_FRAMES,
		'target_track_smoothing': 0.5,
		'target_track_iou_threshold': 0.3,
		'target_track_embedding_threshold': 0.3,
		'target_track_motion_weight': 0.5,
		'target_track_identity_weight': 0.5,
		'target_track_debug': False,
		'adaptive_mask': True,
		'adaptive_mask_feather': 0.2,
		'temporal_mask': True,
		'partial_face_mode': True,
		'target_path': TARGET_PATH
	}

	for state_key, state_value in state_item.items():
		state_manager.init_item(state_key, state_value)

	for module in ( face_classifier, face_detector, face_landmarker, face_recognizer ):
		module.pre_check()


def read_source_frames() -> List[VisionFrame]:
	target_vision_frames : List[VisionFrame] = []

	for frame_number in range(FRAME_TOTAL):
		target_vision_frame = read_static_video_frame(TARGET_PATH, frame_number)

		if target_vision_frame is None:
			break
		target_vision_frames.append(target_vision_frame)
	print('loaded ' + str(len(target_vision_frames)) + ' real video frames')
	return target_vision_frames


def simulate_dropout(target_vision_frames : List[VisionFrame]) -> None:
	# cover the face with a blurred, dark moving patch so the detector genuinely
	# fails for a few frames, the way it does at extreme yaw in real footage
	for dropout_start, dropout_end in DROPOUTS:
		seed_vision_frame = target_vision_frames[max(dropout_start - 1, 0)]
		seed_faces = get_static_faces([ seed_vision_frame ])

		if not seed_faces:
			continue

		seed_face = max(seed_faces, key = lambda face : face.bounding_box[2] - face.bounding_box[0])
		box = seed_face.bounding_box.astype(int)
		center_x = (box[0] + box[2]) // 2
		center_y = (box[1] + box[3]) // 2
		half_size = int(max(box[2] - box[0], box[3] - box[1]) * 0.85)

		for frame_number in range(dropout_start, dropout_end):
			target_vision_frame = target_vision_frames[frame_number]
			frame_height, frame_width = target_vision_frame.shape[:2]
			drift = int((frame_number - dropout_start) * 1.5) - 8
			left = max(center_x + drift - half_size, 0)
			right = min(center_x + drift + half_size, frame_width)
			top = max(center_y - half_size, 0)
			bottom = min(center_y + half_size, frame_height)
			patch = target_vision_frame[top:bottom, left:right]
			patch = cv2.GaussianBlur(patch, (0, 0), 18)
			patch = (patch.astype(numpy.float32) * 0.35).astype(numpy.uint8)
			target_vision_frame[top:bottom, left:right] = patch


def select_window_frames(target_vision_frames : List[VisionFrame], frame_number : int) -> List[VisionFrame]:
	window_frames : List[VisionFrame] = []
	frame_height, frame_width = target_vision_frames[0].shape[:2]

	for window_number in range(frame_number - 2, frame_number + 3):
		if 0 <= window_number < len(target_vision_frames):
			window_frames.append(target_vision_frames[window_number])
		else:
			window_frames.append(numpy.zeros((frame_height, frame_width, 3), dtype = numpy.uint8))
	return window_frames


def build_crop_mask(target_vision_frame : VisionFrame, face : Face) -> tuple:
	frame_height, frame_width = target_vision_frame.shape[:2]
	box = face.bounding_box.astype(int)
	padding = int(max(box[2] - box[0], box[3] - box[1]) * 0.3)
	left = max(box[0] - padding, 0)
	top = max(box[1] - padding, 0)
	right = min(box[2] + padding, frame_width)
	bottom = min(box[3] + padding, frame_height)
	crop_vision_frame = target_vision_frame[top:bottom, left:right]
	crop_height, crop_width = crop_vision_frame.shape[:2]
	crop_mask = numpy.zeros((crop_height, crop_width), dtype = numpy.float32)
	crop_landmark_5 = (face.landmark_set.get('5') - numpy.array([ left, top ])).astype(numpy.float32)
	hull = cv2.convexHull(crop_landmark_5.astype(numpy.int32))
	cv2.fillConvexPoly(crop_mask, hull, 1.0)
	crop_mask = cv2.dilate(crop_mask, numpy.ones((7, 7), dtype = numpy.uint8), iterations = 1)
	return ( left, top, right, bottom ), crop_vision_frame, crop_mask, crop_landmark_5


def draw_hud(panel : VisionFrame, lines : List[str], color : tuple) -> None:
	for index, line in enumerate(lines):
		position_y = 20 + index * 17
		cv2.putText(panel, line, (9, position_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(panel, line, (8, position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def draw_face(panel : VisionFrame, face : Face, color : tuple) -> None:
	box = face.bounding_box.astype(int)
	cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), color, 2)
	for point in face.landmark_set.get('5').astype(int):
		cv2.circle(panel, (point[0], point[1]), 2, color, -1)


def render_frame(target_vision_frame : VisionFrame, old_face : Optional[Face], new_face : Optional[Face], status : TrackingStatus, mask_mass : float, pose_text : str) -> VisionFrame:
	old_panel = cv2.resize(target_vision_frame, (PANEL_WIDTH, PANEL_HEIGHT))
	new_panel = cv2.resize(target_vision_frame, (PANEL_WIDTH, PANEL_HEIGHT))

	if old_face is not None:
		scale_x = PANEL_WIDTH / target_vision_frame.shape[1]
		scale_y = PANEL_HEIGHT / target_vision_frame.shape[0]
		scaled_face = old_face._replace(
			bounding_box = old_face.bounding_box * numpy.array([ scale_x, scale_y, scale_x, scale_y ]),
			landmark_set = { key : value * numpy.array([ scale_x, scale_y ]) for key, value in old_face.landmark_set.items() }
		)
		draw_face(old_panel, scaled_face, (60, 60, 235))
		draw_hud(old_panel, [ 'TARGET FOUND', 'selector: per frame' ], (60, 60, 235))
	else:
		draw_hud(old_panel, [ 'TARGET LOST', 'no face detected', 'swap is NOT applied', 'original face stays' ], (60, 60, 235))

	if new_face is not None:
		scale_x = PANEL_WIDTH / target_vision_frame.shape[1]
		scale_y = PANEL_HEIGHT / target_vision_frame.shape[0]
		scaled_face = new_face._replace(
			bounding_box = new_face.bounding_box * numpy.array([ scale_x, scale_y, scale_x, scale_y ]),
			landmark_set = { key : value * numpy.array([ scale_x, scale_y ]) for key, value in new_face.landmark_set.items() }
		)
		draw_face(new_panel, scaled_face, (90, 230, 90))
		draw_hud(new_panel, [ 'status: ' + status, pose_text, 'identity lock: ON', 'mask mass: ' + str(round(mask_mass)) ], STATUS_COLOR.get(status, (90, 230, 90)))
	else:
		draw_hud(new_panel, [ 'status: ' + status, 'target released', 'after ' + str(MAX_LOST_FRAMES) + ' lost frames' ], STATUS_COLOR.get(status, (90, 230, 90)))

	return numpy.hstack([ old_panel, new_panel ])


def draw_header() -> VisionFrame:
	header = numpy.zeros((HEADER_HEIGHT, PANEL_WIDTH * 2, 3), dtype = numpy.uint8)
	header[:] = (28, 28, 32)
	cv2.putText(header, OLD_TITLE, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 225), 1, cv2.LINE_AA)
	cv2.putText(header, NEW_TITLE, (PANEL_WIDTH + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 120), 1, cv2.LINE_AA)
	cv2.line(header, (PANEL_WIDTH, 0), (PANEL_WIDTH, HEADER_HEIGHT), (70, 70, 70), 1)
	return header


def draw_footer(old_found : List[int], statuses : List[TrackingStatus], current_frame : int) -> VisionFrame:
	footer = numpy.zeros((FOOTER_HEIGHT, PANEL_WIDTH * 2, 3), dtype = numpy.uint8)
	footer[:] = (22, 22, 26)
	cv2.putText(footer, 'OLD', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
	cv2.putText(footer, 'NEW', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

	for frame_number, found in enumerate(old_found):
		color = (90, 200, 90) if found else (60, 60, 235)
		position_x = 50 + frame_number * TICK_WIDTH
		cv2.rectangle(footer, (position_x, 12), (position_x + TICK_WIDTH - 1, 24), color, -1)

	for frame_number, status in enumerate(statuses):
		color = STATUS_COLOR.get(status, (120, 120, 120))
		position_x = 50 + frame_number * TICK_WIDTH
		cv2.rectangle(footer, (position_x, 34), (position_x + TICK_WIDTH - 1, 46), color, -1)

	for dropout_start, dropout_end in DROPOUTS:
		position_start = 50 + dropout_start * TICK_WIDTH
		position_end = 50 + dropout_end * TICK_WIDTH
		cv2.line(footer, (position_start, 6), (position_start, 50), (255, 255, 255), 1)
		cv2.line(footer, (position_end, 6), (position_end, 50), (255, 255, 255), 1)

	marker_x = 50 + current_frame * TICK_WIDTH
	cv2.line(footer, (marker_x, 4), (marker_x, 50), (60, 220, 255), 2)
	return footer


def render_chunk(start_frame : int, end_frame : int) -> None:
	init_state()
	target_vision_frames = read_source_frames()
	simulate_dropout(target_vision_frames)
	source_vision_frames = read_static_images([ SOURCE_PATH ])
	reference_vision_frame = target_vision_frames[0]
	metrics_path = os.path.join(HERE, 'metrics.csv')

	if start_frame == 0:
		with open(metrics_path, 'w', newline = '') as metrics_file:
			writer = csv.writer(metrics_file)
			writer.writerow([ 'frame', 'old_found', 'new_status', 'pose', 'mask_mass', 'identity' ])

	start_time = time.time()

	for frame_number in range(start_frame, end_frame):
		target_vision_frame = target_vision_frames[frame_number]
		window_frames = select_window_frames(target_vision_frames, frame_number)

		target_tracker.set_current_frame_number(frame_number)
		new_faces = target_tracker.select_target_face(reference_vision_frame, source_vision_frames, window_frames)
		status = target_tracker.get_tracking_status()
		new_face = new_faces[0] if new_faces else None

		old_faces = select_faces(reference_vision_frame, source_vision_frames, window_frames)
		old_face = old_faces[0] if old_faces else None

		mask_mass = 0.0
		pose_text = 'pose: n/a'

		if new_face is not None:
			yaw, pitch, roll = face_masker.estimate_head_pose(new_face.landmark_set.get('5'))
			pose_text = 'pose y/p/r: ' + str(round(yaw)) + '/' + str(round(pitch)) + '/' + str(round(roll))
			_, crop_vision_frame, crop_mask, crop_landmark_5 = build_crop_mask(target_vision_frame, new_face)
			adaptive_mask = adaptive_masker.create_adaptive_crop_mask(crop_vision_frame, crop_mask, crop_landmark_5, status)
			mask_mass = float(adaptive_mask.sum())

		state = target_tracker.get_target_track_state()
		identity = float(state.get('tracking_confidence')) if state else 0.0

		composed = numpy.vstack([ draw_header(), render_frame(target_vision_frame, old_face, new_face, status, mask_mass, pose_text) ])
		cv2.imwrite(os.path.join(HERE, 'frames_out', 'frame-' + str(frame_number).zfill(3) + '.png'), composed)

		with open(metrics_path, 'a', newline = '') as metrics_file:
			writer = csv.writer(metrics_file)
			writer.writerow([ frame_number, int(old_face is not None), status, pose_text, round(mask_mass), round(identity, 3) ])

		if frame_number % 20 == 0:
			print('frame ' + str(frame_number) + ' status=' + status + ' old_found=' + str(old_face is not None))

	print('rendered ' + str(start_frame) + '..' + str(end_frame) + ' in ' + str(round(time.time() - start_time)) + 's')


def read_metrics() -> Dict[str, List]:
	metrics : Dict[str, List] =\
	{
		'frame': [],
		'old_found': [],
		'new_status': [],
		'pose': [],
		'mask_mass': [],
		'identity': []
	}

	with open(os.path.join(HERE, 'metrics.csv'), newline = '') as metrics_file:
		reader = csv.reader(metrics_file)
		next(reader, None)

		for row in reader:
			if len(row) < 6:
				continue
			metrics['frame'].append(int(row[0]))
			metrics['old_found'].append(int(row[1]))
			metrics['new_status'].append(row[2])
			metrics['pose'].append(row[3])
			metrics['mask_mass'].append(float(row[4]))
			metrics['identity'].append(float(row[5]))
	return metrics


def assemble() -> None:
	metrics = read_metrics()
	frame_total = len(metrics['frame'])
	output_frames_directory = os.path.join(HERE, 'frames_ready')

	if os.path.isdir(output_frames_directory):
		shutil.rmtree(output_frames_directory)
	os.makedirs(output_frames_directory, exist_ok = True)

	for frame_number in range(frame_total):
		composed = cv2.imread(os.path.join(HERE, 'frames_out', 'frame-' + str(frame_number).zfill(3) + '.png'))
		footer = draw_footer(metrics['old_found'], metrics['new_status'], frame_number)
		cv2.imwrite(os.path.join(output_frames_directory, 'ready-' + str(frame_number).zfill(3) + '.png'), numpy.vstack([ composed, footer ]))

	subprocess.run(
	[
		'ffmpeg', '-y', '-loglevel', 'error',
		'-framerate', str(FRAME_RATE),
		'-i', os.path.join(output_frames_directory, 'ready-%03d.png'),
		'-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
		os.path.join(HERE, 'target_track_demo.mp4')
	], check = True)
	write_report(metrics)


def write_report(metrics : Dict[str, List]) -> None:
	frame_total = len(metrics['frame'])
	old_missing = [ frame for frame, found in zip(metrics['frame'], metrics['old_found']) if not found ]
	new_lost = [ frame for frame, status in zip(metrics['frame'], metrics['new_status']) if status == 'lost' ]
	status_total : Dict[str, int] = {}

	for status in metrics['new_status']:
		status_total[status] = status_total.get(status, 0) + 1
	dropout_lines : List[str] = []

	for dropout_start, dropout_end in DROPOUTS:
		dropout_total = dropout_end - dropout_start
		old_missed = len([ frame for frame in old_missing if dropout_start <= frame < dropout_end ])
		new_held = len([ frame for frame in range(dropout_start, dropout_end) if metrics['new_status'][frame] in ( 'tracked', 'predicted', 'reacquiring' ) ])
		dropout_lines.append('  boşluq ' + str(dropout_start) + '..' + str(dropout_end - 1) + ' (' + str(dropout_total) + ' kadr):')
		dropout_lines.append('    KÖHNƏ seçici hədəfi tapdı: ' + str(dropout_total - old_missed) + '/' + str(dropout_total) + ' (qalan kadrlarda swap yoxdur, orijinal üz qalır)')
		dropout_lines.append('    YENİ izləyici kilidi saxladı: ' + str(new_held) + '/' + str(dropout_total))

	lines =\
	[
		'Target tracking demo - ' + os.path.basename(TARGET_PATH),
		'render edilmiş kadr: ' + str(frame_total),
		'',
		'KÖHNƏ seçici (hər kadr yenidən seçir):',
		'  hədəf tapılmayan kadrlar: ' + str(len(old_missing)) + ' -> ' + str(old_missing),
		'Aşağıdakı kadrlarda swap tətbiq oluna bilmir, ona görə orijinal üz qalır.',
		'',
		'YENİ davamlı izləyici:',
		'  status bölgüsü: ' + str(status_total),
		'  lost olan kadrlar: ' + str(new_lost),
		'',
		'Süni deteksiya boşluqları (üz müvəqqəti örtülüb) - max_lost_frames = ' + str(MAX_LOST_FRAMES) + ':',
		* dropout_lines,
		'',
		'QEYD: KÖHNƏ seçici yalnız pəncərənin orta kadrına baxır',
		'(get_static_faces([ get_middle(target_vision_frames) ])),',
		'ona görə üz görünməyəndə hədəfi tamamilə itirir.',
		'YENİ izləyici əvvəlki müşahidələrdən proqnoz verib kilidi saxlayır,',
		'yalnız ' + str(MAX_LOST_FRAMES) + ' kadrdan çox itkidən sonra hədəfi buraxır.',
		''
	]

	with open(os.path.join(HERE, 'target_track_report.txt'), 'w', encoding = 'utf-8') as report_file:
		report_file.write('\n'.join(lines))
	print('\n'.join(lines))


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', type = int, default = 0)
	parser.add_argument('--end', type = int, default = FRAME_TOTAL)
	parser.add_argument('--assemble-only', action = 'store_true')
	arguments = parser.parse_args()
	os.makedirs(os.path.join(HERE, 'frames_out'), exist_ok = True)

	if arguments.assemble_only:
		assemble()
	else:
		render_chunk(arguments.start, arguments.end)


if __name__ == '__main__':
	main()
