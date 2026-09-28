"""
facefusionswapper.py
====================
FaceFusion 3.9.0 üz dəyişdirmə (face swap) modulu.
GPU (CUDA) dəstəyi və sabit video izləmə (tracking) ilə.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

FACEFUSION_DIR = os.environ.get("FACEFUSION_DIR", "/facefusion")


class FaceFusionError(RuntimeError):
    """FaceFusion prosesi 0-dan fərqli kodla bitdikdə atılır."""


class FaceFusionSwapper:
    """
    FaceFusion 3.9.0 headless-run ətrafında nazik, etibarlı qabıq (wrapper).
    Bütün əlavə parametrlər (options dict) birbaşa CLI flag-lərinə çevrilir.
    """

    def __init__(
        self,
        facefusion_dir: str = FACEFUSION_DIR,
        python_executable: Optional[str] = None,
    ) -> None:
        self.facefusion_dir = Path(facefusion_dir)
        if not (self.facefusion_dir / "facefusion.py").exists():
            raise FileNotFoundError(
                f"facefusion.py tapılmadı: {self.facefusion_dir}. "
                "Dockerfile-in FACEFUSION_DIR dəyişənini yoxlayın."
            )
        self.python = python_executable or shutil.which("python") or "python"

    @staticmethod
    def _options_to_args(options: Optional[dict[str, Any]]) -> list[str]:
        args: list[str] = []
        if not options:
            return args
        for key, value in options.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            if value is True:
                args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *[str(v) for v in value]])
            else:
                args.extend([flag, str(value)])
        return args

    def build_command(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        face_swapper_model: str = "hyperswap_1a_256",
        face_selector_mode: str = "reference",
        execution_thread_count: int = 8,
        face_landmarker_model: str = "hrffa",  # 👈 DEFAULT YENİLƏNDİ (hrffa)
        options: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        cmd = [
            self.python,
            "facefusion.py",
            "headless-run",
            "--source-paths", source_path,
            "--target-path", target_path,
            "--output-path", output_path,
            "--face-selector-mode", face_selector_mode,
            "--execution-providers", "cuda",
            "--execution-thread-count", str(execution_thread_count),
        ]
        cmd.extend(self._options_to_args(options))
        joined = " ".join(cmd)
        if "--face-swapper-model" not in joined:
            cmd.extend(["--face-swapper-model", face_swapper_model])
        if "--face-landmarker-model" not in joined:
            cmd.extend(["--face-landmarker-model", face_landmarker_model])
        return cmd

    def _base_env(self) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        return env

    def swap(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        **kwargs,
    ) -> str:
        cmd = self.build_command(source_path, target_path, output_path, **kwargs)
        process = subprocess.run(
            cmd, cwd=str(self.facefusion_dir), capture_output=True,
            text=True, env=self._base_env(),
        )
        if process.returncode != 0 or not Path(output_path).exists():
            raise FaceFusionError(
                "FaceFusion uğursuz oldu "
                f"(exit code {process.returncode}):\n"
                f"{(process.stderr or process.stdout or '')[-2000:]}"
            )
        return output_path

    def swap_stream(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        **kwargs,
    ):
        cmd = self.build_command(source_path, target_path, output_path, **kwargs)
        process = subprocess.Popen(
            cmd,
            cwd=str(self.facefusion_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=self._base_env(),
        )

        # FaceFusion irəliləyişi tqdm ilə \r ilə yeniləyir; yalnız \n oxusan
        # yükləmə/proses boyu heç nə görmürsən. Həm \r həm \n üzrə bölürük.
        line_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        def _reader() -> None:
            buffer = b""
            stream = process.stdout
            while True:
                chunk = stream.read(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                buffer += chunk
                parts = re.split(b"[\r\n]", buffer)
                buffer = parts[-1]
                for part in parts[:-1]:
                    text = part.decode("utf-8", errors="replace").strip()
                    if text:
                        line_queue.put(text)
            tail = buffer.decode("utf-8", errors="replace").strip()
            if tail:
                line_queue.put(tail)
            line_queue.put(None)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        logs: list[str] = []
        last_yield_time = time.time()
        start_hint = "⏳ FaceFusion başlayır (modellər hazırlanır) — ilk saniyələr sakitdir..."
        while True:
            try:
                line = line_queue.get(timeout=0.3)
            except queue.Empty:
                # Heartbeat: baglanti kesilmasin deye 2 san barda magiza
                # yeni mesaj gondərir (HF proxy boş stream-ləri kəsir).
                if time.time() - last_yield_time >= 2.0:
                    if logs:
                        yield "\n".join(logs[-120:]), False, None
                    else:
                        yield start_hint, False, None
                    last_yield_time = time.time()
                continue
            if line is None:
                break
            logs.append(line)
            if len(logs) > 500:
                logs = logs[-500:]
            now = time.time()
            if now - last_yield_time >= 0.3:
                yield "\n".join(logs[-120:]), False, None
                last_yield_time = now

        reader_thread.join()
        process.wait()

        full_log = "\n".join(logs[-120:])

        if process.returncode != 0 or not Path(output_path).exists():
            yield full_log + f"\n[ERROR] exit code {process.returncode}", False, None
        else:
            output_size = Path(output_path).stat().st_size
            if output_size < 10_000:
                yield full_log + (
                    f"\n[ERROR] Output fayl çox kiçikdir ({output_size} bayt) "
                    "- ehtimal ki, qara ekrandır."
                ), False, None
            else:
                yield full_log, True, output_path
