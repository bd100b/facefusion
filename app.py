"""
app.py
======
Hugging Face Space üçün FaceFusion 3.9.0 — Dinamik model optimizasiyası ilə.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import gradio as gr

from facefusionswapper import FaceFusionSwapper, FaceFusionError


def _log_execution_environment() -> None:
	try:
		import onnxruntime

		print(f"[diag] onnxruntime {onnxruntime.__version__} | device: {onnxruntime.get_device()}")
		print(f"[diag] available providers: {onnxruntime.get_available_providers()}")
	except Exception as exc:
		print(f"[diag] onnxruntime yoxlanila bilmedi: {exc}")


_log_execution_environment()

swapper = FaceFusionSwapper()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TMP_DIR = Path("/app/facefusion_tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Eyni anda yalnız bir swap prosesi: ikinci klik queue-da gözləyib
# spinner göstərməsin, əvəzində dərhal xəbərdarlıq verək.
BUSY = {"active": False, "message": ""}

BASE_MASK_REGIONS = [
    "skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye",
    "glasses", "nose",
]
MOUTH_REGIONS = ["mouth", "upper-lip", "lower-lip"]



def _probe_fps(path: str):
    """Hədəf videonun real fps-ini ölç (VFR videolarda metadata yalan deyə)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        num, den = out.split("/")
        fps = round(int(num) / max(int(den), 1))
        return max(1, min(60, fps))
    except Exception:
        return None


def _save_upload(upload_path: str, prefix: str) -> str:
    ext = Path(upload_path).suffix.lower()
    if not ext:
        ext = ".png" if prefix == "source" else ".mp4"
    dest = TMP_DIR / f"{prefix}{ext}"
    shutil.copyfile(upload_path, dest)
    return str(dest)


def _output_path_for(target_path: str) -> str:
    ext = Path(target_path).suffix.lower()
    if ext not in VIDEO_EXTENSIONS and ext not in IMAGE_EXTENSIONS:
        ext = ".mp4"
    return str(TMP_DIR / f"output{ext}")


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def run_swap(
    source_file, target_file, model, pixel_boost, face_swapper_weight, selector_mode, selector_order,
    landmarker, face_detector_model, face_detector_score, face_landmarker_score,
    face_tracker_score, reference_face_distance, detector_angles,
    proc_face_enhancer, enhancer_model, enhancer_blend,
    proc_expression_restorer, expression_restorer_model, video_preset,
    swap_mouth,
    face_mask_blur, face_mask_padding,
    proc_diffusion, diffusion_strength, diffusion_steps, diffusion_scale,
    staged_source, staged_target,
):
    if source_file is None or target_file is None:
        yield "⚠️ Zəhmət olmasa həm mənbə (üz) həm də hədəf fayl yükləyin.", None, None, None
        return
    if BUSY["active"]:
        yield (
            "⏳ Bir proses artıq işləyir: " + BUSY["message"] + "\n"
            "Zəhmət olmasa bitənə qədər gözləyin (Log qutusu canlı yenilənir).",
            None, None, None,
        )
        return
    BUSY["active"] = True
    BUSY["message"] = "başladıldı..."

    try:
        yield from _run_swap_impl(
            source_file, target_file, model, pixel_boost, face_swapper_weight, selector_mode, selector_order,
            landmarker, face_detector_model, face_detector_score, face_landmarker_score,
            face_tracker_score, reference_face_distance, detector_angles,
            proc_face_enhancer, enhancer_model, enhancer_blend,
            proc_expression_restorer, expression_restorer_model, video_preset,
            swap_mouth,
            face_mask_blur, face_mask_padding,
            proc_diffusion, diffusion_strength, diffusion_steps, diffusion_scale,
            staged_source, staged_target,
        )
    finally:
        BUSY["active"] = False


def _run_swap_impl(
    source_file, target_file, model, pixel_boost, face_swapper_weight, selector_mode, selector_order,
    landmarker, face_detector_model, face_detector_score, face_landmarker_score,
    face_tracker_score, reference_face_distance, detector_angles,
    proc_face_enhancer, enhancer_model, enhancer_blend,
    proc_expression_restorer, expression_restorer_model, video_preset,
    swap_mouth,
    face_mask_blur, face_mask_padding,
    proc_diffusion, diffusion_strength, diffusion_steps, diffusion_scale,
    staged_source, staged_target,
):

    try:
        source_path = staged_source or _save_upload(source_file, "source")
        target_path = staged_target or _save_upload(target_file, "target")
    except OSError as exc:
        yield (
            "⚠️ Yüklənən fayl artıq mövcud deyil. Zəhmət olmasa faylı yenidən "
            f"seçin və bir daha 'Başlat' edin. ({exc})",
            None, None, None,
        )
        return

    output_path = _output_path_for(target_path)
    is_video = _is_video(target_path)

    if Path(output_path).exists():
        os.remove(output_path)

    keep = {Path(source_path).name, Path(target_path).name, Path(output_path).name}
    for f in TMP_DIR.glob("*"):
        if f.name not in keep and f.is_file():
            try:
                f.unlink()
            except OSError:
                pass

    processors = ["face_swapper"]
    # Zencir sirasi onemlidir: swap -> diffusion (realliq berpasi)
    # -> enhancer (son deqiqllesme). Diffusion enhancer-den ONCE gelmelidir.
    if proc_diffusion:
        processors.append("diffusion_swapper")
    if proc_face_enhancer:
        processors.append("face_enhancer")
    if proc_expression_restorer and is_video:
        processors.append("expression_restorer")

    face_mask_types = ["box", "occlusion", "region"]
    if proc_diffusion:
        face_mask_types.append("3d")

    face_mask_regions = list(BASE_MASK_REGIONS)
    if swap_mouth:
        face_mask_regions.extend(MOUTH_REGIONS)

    try:
        padding_list = [int(p) for p in face_mask_padding.split()]
        if len(padding_list) != 4:
            padding_list = [10, 20, 10, 20]
    except Exception:
        padding_list = [10, 20, 10, 20]

    angles = list(detector_angles) if detector_angles else ["0"]
    angles = [str(a).replace("°", "") for a in angles]

    options: dict = {
        "processors": processors,
        "face_swapper_model": model,
        "face_swapper_pixel_boost": pixel_boost if pixel_boost != "None" else None,
        "face_swapper_weight": float(face_swapper_weight) if model == "alphaface_256" else 1.0,
        "face_selector_order": selector_order,
        "reference_face_distance": float(reference_face_distance),
        "face_landmarker_model": landmarker,
        "face_detector_model": face_detector_model,
        "face_detector_size": "640x640",
        "face_detector_angles": angles,
        "face_detector_score": float(face_detector_score),
        "face_landmarker_score": float(face_landmarker_score),
        "face_tracker_score": float(face_tracker_score),
        "face_enhancer_model": enhancer_model,
        "face_enhancer_blend": int(enhancer_blend),
        "expression_restorer_model": expression_restorer_model,
        "face_occluder_model": "xseg_3",
        "face_mask_types": face_mask_types,
        "face_mask_regions": face_mask_regions,
        "face_mask_blur": float(face_mask_blur),
        "face_mask_padding": padding_list,
        "execution_device_ids": [0],
        "output_video_encoder": "libx264",
        "output_audio_encoder": "aac",
        "output_video_fps": _probe_fps(target_path) if is_video else None,
        "output_video_preset": video_preset,
        "output_video_quality": 95,
        "workflow_strategy": "disk",
        "video_memory_strategy": "strict",
        "diffusion_swapper_model": "sd_1_5_inpainting",
        "diffusion_swapper_strength": float(diffusion_strength),
        "diffusion_swapper_steps": int(diffusion_steps),
        "diffusion_swapper_scale": float(diffusion_scale),
    }

    diffusion_note = (
        f" | Diffusion: {diffusion_strength} qüvvə, {diffusion_steps} addım"
        if proc_diffusion else ""
    )
    yield (
        "🚀 FaceFusion prosesi başladılır (Thread: 2, memory: strict)...\n"
        f"Detector: {face_detector_model} | Model: {model} | Pixel Boost: {pixel_boost}{diffusion_note}\n",
        None, None, None,
    )

    try:
        for log, done, result in swapper.swap_stream(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            face_selector_mode=selector_mode,
            execution_thread_count=2,
            options=options,
        ):
            if not done:
                first_line = log.splitlines()[-1] if log else ""
                BUSY["message"] = first_line[:80]
            if done:
                if not Path(result).exists() or Path(result).stat().st_size < 10000:
                    yield log + "\n❌ Xəta: Output fayl boşdur.", None, None, None
                    return
                if _is_video(result):
                    yield (
                        log + "\n✅ Hazırdır!",
                        result,
                        gr.update(value=None, visible=False),
                        gr.update(value=result, visible=True),
                    )
                else:
                    yield (
                        log + "\n✅ Hazırdır!",
                        result,
                        gr.update(value=result, visible=True),
                        gr.update(value=None, visible=False),
                    )
                return
            yield log, None, None, None
    except FaceFusionError as exc:
        yield f"❌ Xəta baş verdi:\n{exc}", None, None, None
    except Exception as exc:
        yield f"❌ Gözlənilməz xəta: {exc}", None, None, None


with gr.Blocks(title="FaceFusion Pro — Face Swap") as demo:
    gr.Markdown(
        "# 🎭 FaceFusion — Professional Üz Dəyişdirmə\n"
        "✨ Dinamik model optimizasiyası və ağıllı tənzimləmə ilə."
    )

    with gr.Row():
        with gr.Column(scale=1):
            source_input = gr.File(
                label="1️⃣ Mənbə üz (Şəkil)",
                file_types=["image"], type="filepath",
            )
            target_input = gr.File(
                label="2️⃣ Hədəf (Şəkil və ya Video)",
                file_types=["image", "video"], type="filepath",
            )

            # Gradio-nun /tmp/gradio temp fayllari eventler arasinda siline
            # bilir (HF Space) -> fayli yukleme ANINDA oz qovlugumuza
            # kopyalayiriq ve klik handler-e gr.State ile otururuk.
            source_staged = gr.State(None)
            target_staged = gr.State(None)

            def _stage_source(upload_path):
                if upload_path is None:
                    return None
                try:
                    return _save_upload(upload_path, "source")
                except OSError:
                    return None

            def _stage_target(upload_path):
                if upload_path is None:
                    return None
                try:
                    return _save_upload(upload_path, "target")
                except OSError:
                    return None

            source_input.upload(
                fn=_stage_source, inputs=[source_input], outputs=[source_staged],
            )
            target_input.upload(
                fn=_stage_target, inputs=[target_input], outputs=[target_staged],
            )

            with gr.Tabs():
                with gr.TabItem("🖼️ Şəkil Ayarları"):
                    model = gr.Dropdown(
                        label="Face Swapper modeli",
                        choices=["hyperswap_1a_256", "alphaface_256", "inswapper_128", "hyperswap_1c_256"],
                        value="hyperswap_1a_256",
                    )
                    pixel_boost = gr.Dropdown(
                        label="Pixel Boost (Yüksək Detal)",
                        choices=["512x512", "256x256", "None"],
                        value="512x512",
                    )
                    
                    # 👈 AlphaFace üçün xüsusi çəki slider-i (başlanğıcda gizli)
                    face_swapper_weight = gr.Slider(
                        label="AlphaFace Weight (Uyğunlaşma çəkisi: 0.85 - 0.90 ideal)",
                        minimum=0.0, maximum=1.0, value=0.90, step=0.05,
                        visible=False,
                    )

                    with gr.Row():
                        proc_face_enhancer = gr.Checkbox(label="Face Enhancer", value=True)
                        enhancer_model = gr.Dropdown(
                            label="Enhancer Modeli",
                            choices=["gfpgan_1.4", "codeformer"],
                            value="codeformer",
                        )
                    enhancer_blend = gr.Slider(
                        label="Enhancer Blend (100 = tam gücləndirmə)",
                        minimum=0, maximum=100, value=80, step=5,
                    )
                    with gr.Row():
                        selector_mode = gr.Dropdown(
                            label="Üz seçim rejimi",
                            choices=["reference", "one", "many"],
                            value="reference",
                        )
                        selector_order = gr.Dropdown(
                            label="Üz seçim sırası",
                            choices=["large-small", "small-large", "best-worst", "worst-best", "left-right", "right-left"],
                            value="large-small",
                        )
                    reference_face_distance = gr.Slider(
                        label="Reference Face Distance (0.3 ideal)",
                        minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                    )

                with gr.TabItem("🎬 Video Ayarları"):
                    video_preset = gr.Dropdown(
                        label="Video Output Preset",
                        choices=["ultrafast", "fast", "medium", "slow"],
                        value="medium",
                    )
                    with gr.Row():
                        proc_expression_restorer = gr.Checkbox(
                            label="Expression Restorer (mimika qoruma)", value=True,
                        )
                        expression_restorer_model = gr.Dropdown(
                            label="Expression Restorer Modeli",
                            choices=["live_portrait"],
                            value="live_portrait",
                        )
                    face_tracker_score = gr.Slider(
                        label="Face Tracker Score",
                        minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                    )

                with gr.TabItem("⚙️ Detektor & Maska"):
                    with gr.Row():
                        face_detector_model = gr.Dropdown(
                            label="Face Detector Model",
                            choices=["scrfd", "retinaface", "yolo_face", "yunet"],
                            value="yolo_face",
                        )
                        landmarker = gr.Dropdown(
                            label="Face Landmarker",
                            choices=["hrffa", "2dfan4", "peppa_wutz"],
                            value="hrffa",
                        )
                    face_detector_score = gr.Slider(
                        label="Face Detector Score",
                        minimum=0.1, maximum=0.8, value=0.5, step=0.05,
                    )
                    face_landmarker_score = gr.Slider(
                        label="Face Landmarker Score",
                        minimum=0.1, maximum=1.0, value=0.50, step=0.05,
                    )
                    detector_angles = gr.CheckboxGroup(
                        label="Detector Bucaqları",
                        choices=["0", "90", "180", "270", "330", "30"],
                        value=["0", "90", "180", "270"],
                    )
                    swap_mouth = gr.Checkbox(
                        label="Ağız/Dodaqları da dəyişdir", value=False,
                    )
                    face_mask_blur = gr.Slider(
                        label="Face Mask Blur",
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    )
                    face_mask_padding = gr.Textbox(
                        label="Face Mask Padding (Üst Sağ Alt Sol)",
                        value="15 25 15 25",
                    )

                    gr.Markdown("#### 🌀 Diffusion Regenerasiya (extreme bucaqlar üçün)")
                    proc_diffusion = gr.Checkbox(
                        label="Diffusion Swapper aktiv et (polar / arxaya dönük baş)",
                        value=False,
                    )
                    with gr.Row():
                        diffusion_strength = gr.Slider(
                            label="Strength (0.55 standart, 0.6–0.75 extreme)",
                            minimum=0.1, maximum=1.0, value=0.55, step=0.05,
                        )
                        diffusion_steps = gr.Slider(
                            label="Addım sayı (24 standart)",
                            minimum=8, maximum=64, value=24, step=1,
                        )
                    diffusion_scale = gr.Slider(
                        label="Identity Scale (yüksək = orijinala daha bağlı)",
                        minimum=1.0, maximum=10.0, value=4.5, step=0.5,
                    )

                run_btn = gr.Button("🚀 Başlat", variant="primary", size="lg")

        with gr.Column(scale=1):
            log_box = gr.Textbox(label="📝 Sistem Logları", lines=8, max_lines=12, interactive=False)
            gr.Markdown("### 📤 Nəticə")
            preview_image = gr.Image(label="Şəkil nəticəsi", visible=False)
            preview_video = gr.Video(label="Video nəticəsi", visible=False)
            result_file = gr.File(label="Faylı Yüklə", interactive=False)

    # 🧠 DİNAMİK MƏNTİQ: Model dəyişəndə avtomatik tənzimləmə
    def on_model_change(selected_model):
        if selected_model == "alphaface_256":
            return gr.update(value="512x512"), gr.update(visible=True)
        else:
            return gr.update(), gr.update(visible=False)

    model.change(
        fn=on_model_change,
        inputs=[model],
        outputs=[pixel_boost, face_swapper_weight],
    )

    run_btn.click(
        fn=run_swap,
        inputs=[
            source_input, target_input, model, pixel_boost, face_swapper_weight, selector_mode, selector_order,
            landmarker, face_detector_model, face_detector_score, face_landmarker_score,
            face_tracker_score, reference_face_distance, detector_angles,
            proc_face_enhancer, enhancer_model, enhancer_blend,
            proc_expression_restorer, expression_restorer_model, video_preset,
            swap_mouth,
            face_mask_blur, face_mask_padding,
            proc_diffusion, diffusion_strength, diffusion_steps, diffusion_scale,
            source_staged, target_staged,
        ],
        outputs=[log_box, result_file, preview_image, preview_video],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(TMP_DIR)],
    )
