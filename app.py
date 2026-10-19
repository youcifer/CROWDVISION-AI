import os
import time
import uuid
import math
import shutil
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, abort
import cv2
import numpy as np
import imageio_ffmpeg
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1 GB max upload for long clips
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'uploads')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'sample')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Ensure protest tracker configuration file exists
PROTEST_TRACKER_PATH = os.path.join(BASE_DIR, 'protest_tracker.yaml')
if not os.path.exists(PROTEST_TRACKER_PATH):
    with open(PROTEST_TRACKER_PATH, 'w') as f:
        f.write("""tracker_type: bytetrack
track_high_thresh: 0.22      # Lower threshold to detect people behind signs/flags
track_low_thresh: 0.08       # Second stage ByteTrack association for heavy occlusions
new_track_thresh: 0.28       # Threshold to initiate a new track
track_buffer: 80             # Extended occlusion memory (~3-4 seconds behind banners/flags)
match_thresh: 0.72           # Spatial matching tolerance across frames
fuse_score: True
""")

# Preload YOLO models into memory
print("Loading YOLO models (Nano & Small)...")
MODELS = {
    'yolov8n': YOLO(os.path.join(BASE_DIR, 'yolov8n.pt')),
    'yolov8s': YOLO(os.path.join(BASE_DIR, 'yolov8s.pt'))
}
print("YOLO models ready.")

# In-memory tracking of tasks
TASKS = {}
TASKS_LOCK = threading.Lock()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_color_for_id(track_id: int):
    """Generate a consistent, vibrant BGR color for each track ID using HSV."""
    if track_id is None:
        return (0, 255, 255)
    hue = int((track_id * 47) % 180)
    hsv_pixel = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def apply_clahe(bgr_frame):
    """
    Adaptive Contrast Enhancement (CLAHE) on the Luminance channel.
    Crucial for protest footage: brings out faces and head contours
    under shadows, hats, placards, and low-contrast mobile video compression.
    """
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def draw_hud(frame, current_count, total_unique, current_time_str, fps_val, mode_str="Standard"):
    """Draw a sleek, modern translucent HUD banner onto the video frame."""
    h, w = frame.shape[:2]
    hud_w = min(460, int(w * 0.55))
    hud_h = 76
    margin = 16

    # Create HUD overlay with subtle dark glassmorphism
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + hud_w, margin + hud_h), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Accent cyan / amber border
    border_color = (6, 214, 212) if mode_str == "Standard" else (11, 158, 245)  # Amber for protest mode
    cv2.rectangle(frame, (margin, margin), (margin + hud_w, margin + hud_h), (40, 48, 64), 1)
    cv2.rectangle(frame, (margin, margin), (margin + 4, margin + hud_h), border_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line 1: Live Count & Unique Count
    cv2.putText(frame, "LIVE CROWD:", (margin + 16, margin + 28), font, 0.52, (160, 175, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{current_count:02d}", (margin + 128, margin + 29), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "TOTAL UNIQUE:", (margin + 195, margin + 28), font, 0.52, (160, 175, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{total_unique}", (margin + 330, margin + 29), font, 0.75, border_color, 2, cv2.LINE_AA)

    # Line 2: Time and Processing indicator
    cv2.putText(frame, f"TIME: {current_time_str}", (margin + 16, margin + 60), font, 0.44, (130, 145, 165), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_val:.1f}  |  {mode_str}", (margin + 195, margin + 60), font, 0.44, (120, 140, 160), 1, cv2.LINE_AA)

def draw_person_overlay(frame, x1, y1, x2, y2, track_id, conf, style='dots'):
    """
    Draw detection overlays with specific optimizations for dense crowds:
    In dense protests with signs/flags, head pinpoints and glowing dots
    eliminate bounding box clutter while pinpointing each individual.
    """
    color = get_color_for_id(track_id)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    # Head / upper-chest center (15-20% from top of box)
    head_y = int(y1 + max(6, (y2 - y1) * 0.18))
    head_pt = (cx, head_y)

    if style in ('boxes', 'hybrid'):
        # Sleek bracketed box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        corner_len = max(6, int(min(x2 - x1, y2 - y1) * 0.2))
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

    if style in ('dots', 'hybrid', 'heatmap'):
        # Glowing radar head pinpoint (ideal for packed protest crowds)
        cv2.circle(frame, head_pt, 6, color, -1)
        cv2.circle(frame, head_pt, 11, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, head_pt, 15, color, 1, cv2.LINE_AA)

    # Label text
    id_label = f"#{track_id}" if track_id is not None else "Person"
    label_text = f"{id_label} {int(conf * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Position badge either above box or above head point
    if style == 'dots' or style == 'heatmap':
        pill_x1 = max(0, cx - tw // 2 - 4)
        pill_y1 = max(0, head_y - th - 12)
    else:
        pill_x1 = max(0, x1)
        pill_y1 = max(0, y1 - th - 7)

    pill_x2 = pill_x1 + tw + 8
    pill_y2 = pill_y1 + th + 5

    # Draw label badge
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (12, 16, 26), -1)
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)
    cv2.putText(frame, label_text, (pill_x1 + 4, pill_y2 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def apply_density_heatmap_overlay(frame, head_points, alpha=0.32):
    """
    Renders a dynamic Gaussian crowd density heatmap over the frame.
    Dense protest clusters show up as vibrant red/yellow thermal hotspots!
    """
    if not head_points:
        return frame

    h, w = frame.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    radius = max(24, int(min(w, h) * 0.05))
    for (hx, hy) in head_points:
        if 0 <= hx < w and 0 <= hy < h:
            cv2.circle(density_map, (hx, hy), radius, 1.0, -1)

    # Smooth density with large Gaussian kernel
    kernel_size = (radius * 2 + 1) | 1
    density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), 0)

    # Normalize to 0-255
    max_val = np.max(density_map)
    if max_val > 0:
        norm_map = (density_map / max_val * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Mask only regions with crowd density
        mask = norm_map > 15
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(frame[mask], 1.0 - alpha, color_map[mask], alpha, 0)
        return blended
    return frame

def process_video_task(
    task_id,
    input_path,
    output_path,
    model_choice='yolov8s',
    conf_thresh=0.22,
    iou_thresh=0.55,
    frame_stride=2,
    style='dots',
    protest_mode=True,
    enhance_contrast=True,
    enable_heatmap=False,
    imgsz=960,
    max_dim=1280
):
    """
    Background worker that runs YOLO tracking with protest-specific
    tuning, occlusion buffering, adaptive contrast enhancement, and H.264 streaming.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Downsample dimensions if too large for responsiveness
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / float(max(orig_w, orig_h))
            out_w = int(orig_w * scale)
            out_h = int(orig_h * scale)
            out_w = out_w if out_w % 2 == 0 else out_w - 1
            out_h = out_h if out_h % 2 == 0 else out_h - 1
        else:
            out_w = orig_w if orig_w % 2 == 0 else orig_w - 1
            out_h = orig_h if orig_h % 2 == 0 else orig_h - 1

        stride = max(1, int(frame_stride))
        effective_out_fps = max(5.0, round(orig_fps / stride, 2))

        # Choose model (nano or small)
        selected_model = MODELS.get(model_choice, MODELS['yolov8s'])

        # Tracker config
        tracker_config = PROTEST_TRACKER_PATH if protest_mode else "bytetrack.yaml"

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{out_w}x{out_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(effective_out_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '22',
            '-movflags', '+faststart',
            output_path
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        mode_name = "Protest & Dense Crowd Mode" if protest_mode else "Standard Mode"

        with TASKS_LOCK:
            TASKS[task_id].update({
                'status': 'processing',
                'total_frames': total_frames,
                'orig_fps': orig_fps,
                'out_fps': effective_out_fps,
                'orig_width': orig_w,
                'orig_height': orig_h,
                'out_width': out_w,
                'out_height': out_h,
                'stride': stride,
                'model_used': model_choice,
                'mode_name': mode_name,
                'duration_sec': round(total_frames / orig_fps, 2) if orig_fps > 0 else 0
            })

        unique_ids = set()
        frame_counts = []
        timeline_seconds = {}
        frame_idx = 0
        processed_count = 0
        max_concurrent_count = 0
        peak_frame = 0
        start_time = time.time()

        while True:
            # Check for cancellation
            with TASKS_LOCK:
                if TASKS[task_id].get('status') == 'cancelled':
                    break

            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping for speed optimization
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            if scale != 1.0 or frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Preprocessing: Apply CLAHE if enabled or low quality protest video
            if enhance_contrast:
                detection_input = apply_clahe(frame)
            else:
                detection_input = frame

            # YOLOv8 Tracking with ByteTrack (person class 0)
            results = selected_model.track(
                detection_input,
                persist=True,
                tracker=tracker_config,
                classes=[0],
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                verbose=False
            )

            current_frame_people = 0
            current_boxes = []
            head_points = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                    if track_id is not None:
                        unique_ids.add(track_id)

                    current_boxes.append((x1, y1, x2, y2, track_id, conf))
                    cx = int((x1 + x2) / 2)
                    head_y = int(y1 + max(6, (y2 - y1) * 0.18))
                    head_points.append((cx, head_y))
                    current_frame_people += 1

            # Render Heatmap layer if enabled
            if enable_heatmap or style == 'heatmap':
                frame = apply_density_heatmap_overlay(frame, head_points, alpha=0.35)

            # Draw person overlays
            for (x1, y1, x2, y2, tid, cnf) in current_boxes:
                draw_person_overlay(frame, x1, y1, x2, y2, tid, cnf, style=style)

            # Tracking stats
            frame_counts.append(current_frame_people)
            if current_frame_people > max_concurrent_count:
                max_concurrent_count = current_frame_people
                peak_frame = frame_idx

            # Timeline per second
            curr_sec = math.floor(frame_idx / orig_fps)
            if curr_sec not in timeline_seconds:
                timeline_seconds[curr_sec] = {
                    'counts': [],
                    'unique_at_sec': len(unique_ids)
                }
            timeline_seconds[curr_sec]['counts'].append(current_frame_people)

            # Draw HUD
            cur_time_sec = frame_idx / orig_fps
            time_str = f"{int(cur_time_sec // 60):02d}:{int(cur_time_sec % 60):02d}.{int((cur_time_sec % 1) * 10)}"
            fps_live = processed_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            draw_hud(frame, current_frame_people, len(unique_ids), time_str, fps_live, mode_str=mode_name)

            # Pipe frame to FFmpeg
            ffmpeg_proc.stdin.write(frame.tobytes())

            processed_count += 1
            frame_idx += 1

            # Update progress status periodically
            if processed_count % 3 == 0 or frame_idx >= total_frames:
                elapsed = time.time() - start_time
                progress = min(100.0, round((frame_idx / max(1, total_frames)) * 100, 1))
                remaining_frames = (total_frames - frame_idx) / stride
                est_rem = (remaining_frames / fps_live) if fps_live > 0 else 0

                with TASKS_LOCK:
                    TASKS[task_id].update({
                        'progress': progress,
                        'current_frame': frame_idx,
                        'processed_frames': processed_count,
                        'current_count': current_frame_people,
                        'unique_so_far': len(unique_ids),
                        'fps': round(fps_live, 1),
                        'time_elapsed': round(elapsed, 1),
                        'time_remaining_est': max(0, round(est_rem, 1))
                    })

        # Close video & ffmpeg streams
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=10)
        except Exception as e:
            print("FFmpeg close/wait error:", e)
            try:
                ffmpeg_proc.kill()
            except Exception:
                pass

        # Check if cancelled
        with TASKS_LOCK:
            if TASKS[task_id].get('status') == 'cancelled':
                if os.path.exists(output_path):
                    os.remove(output_path)
                return

        # Prepare summary & analytics
        avg_count = round(float(np.mean(frame_counts)), 2) if frame_counts else 0.0
        duration_sec = round(total_frames / orig_fps, 2) if orig_fps > 0 else 0
        peak_time_sec = round(peak_frame / orig_fps, 1) if orig_fps > 0 else 0
        peak_time_str = f"{int(peak_time_sec // 60):02d}:{int(peak_time_sec % 60):02d}"

        # Build timeline series (sorted by second)
        timeline = []
        for sec in sorted(timeline_seconds.keys()):
            sec_counts = timeline_seconds[sec]['counts']
            sec_avg = round(float(np.mean(sec_counts)), 1)
            sec_max = int(np.max(sec_counts))
            sec_min = int(np.min(sec_counts))
            timeline.append({
                'second': sec,
                'time_str': f"{sec // 60:02d}:{sec % 60:02d}",
                'count': sec_max,
                'avg_count': sec_avg,
                'min_count': sec_min,
                'unique_cumulative': timeline_seconds[sec]['unique_at_sec']
            })

        # Build 5-second interval summary table
        interval_size = 5
        intervals = []
        if timeline:
            max_sec = timeline[-1]['second']
            for start_s in range(0, max_sec + 1, interval_size):
                end_s = min(max_sec, start_s + interval_size - 1)
                chunk_items = [t for t in timeline if start_s <= t['second'] <= end_s]
                if chunk_items:
                    chunk_counts = [t['count'] for t in chunk_items]
                    trend = "steady"
                    if len(chunk_counts) >= 2:
                        diff = chunk_counts[-1] - chunk_counts[0]
                        if diff >= 2:
                            trend = "rising"
                        elif diff <= -2:
                            trend = "falling"
                    intervals.append({
                        'interval_str': f"{start_s // 60:02d}:{start_s % 60:02d} - {end_s // 60:02d}:{end_s % 60:02d}",
                        'min': min(chunk_counts),
                        'max': max(chunk_counts),
                        'avg': round(float(np.mean(chunk_counts)), 1),
                        'trend': trend
                    })

        total_elapsed = round(time.time() - start_time, 1)

        with TASKS_LOCK:
            TASKS[task_id].update({
                'status': 'completed',
                'progress': 100.0,
                'current_frame': total_frames,
                'processed_frames': processed_count,
                'time_elapsed': total_elapsed,
                'summary': {
                    'max_concurrent_count': max_concurrent_count,
                    'peak_time_str': peak_time_str,
                    'peak_time_sec': peak_time_sec,
                    'total_unique_count': len(unique_ids),
                    'avg_count': avg_count,
                    'duration_sec': duration_sec,
                    'total_frames': total_frames,
                    'processed_frames': processed_count,
                    'orig_resolution': f"{orig_w}x{orig_h}",
                    'output_resolution': f"{out_w}x{out_h}",
                    'processing_time_sec': total_elapsed,
                    'effective_fps': round(processed_count / total_elapsed, 1) if total_elapsed > 0 else 0,
                    'model_used': model_choice,
                    'mode_used': mode_name
                },
                'timeline': timeline,
                'intervals': intervals,
                'video_url': f"/api/video/{task_id}"
            })
        print(f"Task {task_id} completed successfully in {total_elapsed}s.")

    except Exception as exc:
        print(f"Error processing task {task_id}: {exc}")
        import traceback
        traceback.print_exc()
        with TASKS_LOCK:
            TASKS[task_id].update({
                'status': 'failed',
                'error': str(exc)
            })

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start background analysis."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Parse options
    protest_mode = request.form.get('protest_mode', 'true').lower() in ('true', '1', 'on')
    model_choice = request.form.get('model_choice', 'yolov8s' if protest_mode else 'yolov8n')
    conf_thresh = float(request.form.get('confidence', 0.22 if protest_mode else 0.35))
    iou_thresh = float(request.form.get('iou', 0.55 if protest_mode else 0.70))
    stride = int(request.form.get('stride', 2))
    style = request.form.get('style', 'dots' if protest_mode else 'hybrid')
    enhance_contrast = request.form.get('enhance_contrast', 'true' if protest_mode else 'false').lower() in ('true', '1', 'on')
    enable_heatmap = request.form.get('heatmap', 'false').lower() in ('true', '1', 'on')
    imgsz = int(request.form.get('imgsz', 960 if protest_mode else 640))

    task_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    input_filename = f"{task_id}_in.{ext}"
    output_filename = f"{task_id}_out.mp4"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    file.save(input_path)

    with TASKS_LOCK:
        TASKS[task_id] = {
            'task_id': task_id,
            'filename': file.filename,
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'output_path': output_path,
            'input_path': input_path
        }

    thread = threading.Thread(
        target=process_video_task,
        args=(
            task_id, input_path, output_path,
            model_choice, conf_thresh, iou_thresh,
            stride, style, protest_mode,
            enhance_contrast, enable_heatmap, imgsz
        ),
        daemon=True
    )
    thread.start()

    return jsonify({
        'task_id': task_id,
        'filename': file.filename,
        'status': 'queued'
    })

@app.route('/api/sample', methods=['POST'])
def trigger_sample():
    """Generate or provide a sample crowd video for testing."""
    sample_src = os.path.join(SAMPLE_DIR, 'sample_crowd.mp4')
    if not os.path.exists(sample_src):
        generate_sample_video(sample_src)

    task_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}_sample.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{task_id}_out.mp4")

    shutil.copyfile(sample_src, input_path)

    with TASKS_LOCK:
        TASKS[task_id] = {
            'task_id': task_id,
            'filename': 'sample_crowd.mp4',
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'output_path': output_path,
            'input_path': input_path
        }

    thread = threading.Thread(
        target=process_video_task,
        args=(
            task_id, input_path, output_path,
            'yolov8s', 0.25, 0.55,
            1, 'dots', True,
            True, False, 960
        ),
        daemon=True
    )
    thread.start()

    return jsonify({
        'task_id': task_id,
        'filename': 'sample_crowd.mp4',
        'status': 'queued'
    })

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Retrieve current processing status or final analysis results."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(task)

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running analysis task."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        task['status'] = 'cancelled'
    return jsonify({'success': True, 'task_id': task_id})

@app.route('/api/video/<task_id>', methods=['GET'])
def get_video(task_id):
    """Serve the processed MP4 video with HTTP partial range support."""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    if not task:
        abort(404)

    output_path = task.get('output_path')
    if not output_path or not os.path.exists(output_path):
        abort(404)

    return send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=False,
        conditional=True
    )

def generate_sample_video(dest_path):
    """Synthesize a sample crowd video."""
    w, h, fps, duration = 854, 480, 24, 8
    total_frames = fps * duration

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'fast', '-crf', '20',
        dest_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    np.random.seed(42)
    pedestrians = []
    for i in range(16):
        direction = 1 if i % 2 == 0 else -1
        speed = float(np.random.uniform(2.5, 4.5)) * direction
        start_x = float(np.random.randint(50, w - 100)) if i < 8 else (float(-50) if direction > 0 else float(w + 50))
        y = float(np.random.randint(220, 380))
        scale = float(0.65 + (y - 220) / 300.0)
        pedestrians.append({
            'x': start_x,
            'y': y,
            'speed': speed,
            'scale': scale,
            'color': (int(np.random.randint(50, 180)), int(np.random.randint(50, 180)), int(np.random.randint(180, 240))),
            'phase': float(np.random.uniform(0, math.pi * 2))
        })

    for f_idx in range(total_frames):
        frame = np.full((h, w, 3), (35, 42, 54), dtype=np.uint8)
        cv2.line(frame, (0, 200), (w, 200), (45, 55, 70), 2)
        cv2.rectangle(frame, (0, 200), (w, h), (40, 48, 62), -1)
        for gx in range(0, w, 90):
            cv2.line(frame, (gx, 200), (int(gx + (gx - w/2)*0.8), h), (48, 58, 74), 1)

        pedestrians.sort(key=lambda p: p['y'])
        for p in pedestrians:
            p['x'] += p['speed']
            if p['speed'] > 0 and p['x'] > w + 60:
                p['x'] = -60
            elif p['speed'] < 0 and p['x'] < -60:
                p['x'] = w + 60

            px = int(p['x'])
            py = int(p['y'])
            sc = p['scale']
            leg_swing = math.sin((f_idx * 0.35) + p['phase']) * 14 * sc

            head_r = int(12 * sc)
            torso_h = int(38 * sc)
            torso_w = int(22 * sc)
            leg_h = int(45 * sc)

            cv2.ellipse(frame, (px, py + int(45 * sc)), (int(20 * sc), int(6 * sc)), 0, 0, 360, (25, 30, 40), -1)
            head_center = (px, py - torso_h - head_r + 4)
            cv2.circle(frame, head_center, head_r, (215, 225, 235), -1)

            torso_top_left = (px - torso_w // 2, py - torso_h)
            torso_bot_right = (px + torso_w // 2, py)
            cv2.rectangle(frame, torso_top_left, torso_bot_right, p['color'], -1)

            cv2.line(frame, (px - 5, py), (int(px - 5 + leg_swing), py + leg_h), (20, 25, 35), max(2, int(4 * sc)))
            cv2.line(frame, (px + 5, py), (int(px + 5 - leg_swing), py + leg_h), (20, 25, 35), max(2, int(4 * sc)))

            cv2.line(frame, (px - torso_w // 2, py - torso_h + 8), (int(px - torso_w // 2 - leg_swing * 0.7), py - 5), (200, 210, 220), max(2, int(3 * sc)))
            cv2.line(frame, (px + torso_w // 2, py - torso_h + 8), (int(px + torso_w // 2 + leg_swing * 0.7), py - 5), (200, 210, 220), max(2, int(3 * sc)))

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

if __name__ == '__main__':
    print("Starting CrowdVision AI Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
