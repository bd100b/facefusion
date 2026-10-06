"""Static server for the demo folder.

`python -m http.server` is single threaded and ignores HTTP range requests, so
while the mp4 is streaming every other request (the page, the report, the
health check) has to wait for it. That stalls the browser until it reports a
lost connection. This server accepts every request on its own thread and
answers range requests so the video stays seekable.
"""

import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE_PATTERN = re.compile(r'\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$')


class RangeRequestHandler(SimpleHTTPRequestHandler):
	remaining_bytes = None

	def send_head(self):
		range_header = self.headers.get('Range')

		if not range_header:
			return super().send_head()

		file_path = self.translate_path(self.path)
		range_match = RANGE_PATTERN.match(range_header)

		if not range_match:
			return super().send_head()

		range_start_text = range_match.group(1)
		range_end_text = range_match.group(2)
		try:
			file_handle = open(file_path, 'rb')
		except OSError:
			self.send_error(404, 'File not found')
			return None

		file_handle.seek(0, 2)
		file_size = file_handle.tell()
		range_start = int(range_start_text) if range_start_text else 0
		range_end = int(range_end_text) if range_end_text else file_size - 1

		if not range_start_text and range_end_text:
			range_start = max(file_size - int(range_end_text), 0)
			range_end = file_size - 1
		range_end = min(range_end, file_size - 1)

		if range_start >= file_size or range_start > range_end:
			file_handle.close()
			self.send_response(416)
			self.send_header('Content-Range', 'bytes */' + str(file_size))
			self.end_headers()
			return None

		file_handle.seek(range_start)
		self.remaining_bytes = range_end - range_start + 1
		self.send_response(206)
		self.send_header('Content-Type', self.guess_type(file_path))
		self.send_header('Content-Length', str(self.remaining_bytes))
		self.send_header('Content-Range', 'bytes ' + str(range_start) + '-' + str(range_end) + '/' + str(file_size))
		self.send_header('Accept-Ranges', 'bytes')
		self.end_headers()
		return file_handle

	def copyfile(self, source, output_file):
		if self.remaining_bytes is None:
			return super().copyfile(source, output_file)

		while self.remaining_bytes > 0:
			chunk = source.read(min(65536, self.remaining_bytes))

			if not chunk:
				break
			output_file.write(chunk)
			self.remaining_bytes -= len(chunk)

	def end_headers(self):
		self.send_header('Cache-Control', 'no-store')
		super().end_headers()


def main() -> None:
	demo_directory = os.path.dirname(os.path.abspath(__file__))
	server_port = int(os.environ.get('PORT', '8000'))
	handler = partial(RangeRequestHandler, directory = demo_directory)
	server = ThreadingHTTPServer(('0.0.0.0', server_port), handler)
	print('serving ' + demo_directory + ' on 0.0.0.0:' + str(server_port))
	server.serve_forever()


if __name__ == '__main__':
	main()
