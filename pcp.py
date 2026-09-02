"""PhotoCherryPick (PCP) v0.3 - extract the best face frame from each scene."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import mediapipe as mp
import piexif
from pymediainfo import MediaInfo
from scenedetect import ContentDetector, SceneManager, open_video


VERSION = "0.3"
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# --- CONFIGURATION ---
SMILE_THRESHOLD = 0.5
EYE_OPEN_THRESHOLD = 0.5
BLUR_THRESHOLD = 100.0
FRAMES_PER_SCENE_CHECK = 15
MAX_FACES = 10

# MediaInfo fields are ordered from the closest to the original capture date
# to container bookkeeping dates.
DATE_FIELDS = ("recorded_date", "mastered_date", "encoded_date", "tagged_date")
MAKE_FIELDS = ("make", "comapplequicktime_make", "com_apple_quicktime_make")
MODEL_FIELDS = ("model", "comapplequicktime_model", "com_apple_quicktime_model")

# MediaPipe Face Landmarker blendshapes used for expression scoring.
REQUIRED_BLENDSHAPES = {
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "mouthSmileLeft",
    "mouthSmileRight",
}


def parse_mediainfo_datetime(value):
    """Parse the ISO-like date strings returned by MediaInfo."""
    if not value:
        return None

    value = str(value).strip()
    if value.startswith("UTC "):
        value = value[4:] + "+00:00"
    elif value.endswith(" UTC"):
        value = value[:-4] + "+00:00"
    elif value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _first_track_value(track, field_names):
    for field_name in field_names:
        value = getattr(track, field_name, None)
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            return str(value)
    return None


def get_video_metadata(video_path):
    """Read capture metadata, falling back explicitly to the file mtime."""
    video_path = Path(video_path)
    captured_at = None
    date_source = None
    make = None
    model = None

    try:
        tracks = MediaInfo.parse(str(video_path)).tracks
    except Exception as exc:
        print(f"Warning: video metadata unavailable ({exc}); using file date.")
        tracks = []

    for track in tracks:
        if track.track_type not in {"General", "Video"}:
            continue

        if captured_at is None:
            for field_name in DATE_FIELDS:
                parsed = parse_mediainfo_datetime(getattr(track, field_name, None))
                if parsed is not None:
                    captured_at = parsed
                    date_source = f"embedded video metadata ({field_name})"
                    break

        make = make or _first_track_value(track, MAKE_FIELDS)
        model = model or _first_track_value(track, MODEL_FIELDS)

    if captured_at is None:
        # A portable file creation time does not exist on every OS. The mtime is
        # therefore the explicit, reproducible fallback stored in UserComment.
        captured_at = datetime.fromtimestamp(video_path.stat().st_mtime).astimezone()
        date_source = "video file modification time (embedded original date unavailable)"

    return {
        "datetime": captured_at,
        "date_source": date_source,
        "make": make,
        "model": model,
        "source": video_path.name,
    }


def format_timestamp(seconds):
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def sample_frame_indices(start_frame, end_frame, sample_count):
    """Return at most sample_count evenly-spaced frames, away from cut edges."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    duration = end_frame - start_frame
    if duration <= 0:
        return []
    if duration <= sample_count:
        return list(range(start_frame, end_frame))

    # Sample the centre of equal-width bins, avoiding transition frames at cuts.
    return [
        start_frame + ((2 * index + 1) * duration) // (2 * sample_count)
        for index in range(sample_count)
    ]


def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_faces(face_blendshapes, smile_threshold, eye_open_threshold):
    """Return worst-face metrics so every detected person must look good."""
    face_metrics = []
    for categories in face_blendshapes:
        scores = {category.category_name: category.score for category in categories}
        if not REQUIRED_BLENDSHAPES.issubset(scores):
            continue

        eye_open = 1.0 - max(scores["eyeBlinkLeft"], scores["eyeBlinkRight"])
        smile = (scores["mouthSmileLeft"] + scores["mouthSmileRight"]) / 2.0
        face_metrics.append((eye_open, smile))

    if not face_metrics:
        return None

    min_eye_open = min(eye_open for eye_open, _ in face_metrics)
    min_smile = min(smile for _, smile in face_metrics)

    # The least successful face determines group-photo quality: one blinking
    # person must not be hidden by another person's high score.
    return {
        "face_count": len(face_metrics),
        "eye_open": min_eye_open,
        "smile": min_smile,
        "score": (min_eye_open + min_smile) / 2.0,
        "perfect": min_eye_open >= eye_open_threshold and min_smile >= smile_threshold,
    }


def candidate_rank(candidate):
    """Perfect frames win, then more detected faces, quality, and sharpness."""
    return (
        candidate["perfect"],
        candidate["face_count"],
        candidate["score"],
        candidate["sharpness"],
    )


def _exif_offset(value):
    offset = value.strftime("%z")
    return f"{offset[:3]}:{offset[3:]}" if offset else None


def save_frame_with_exif(frame, output_path, frame_idx, timestamp_sec, video_metadata):
    """Create a JPEG without overwriting and remove it if EXIF writing fails."""
    output_path = Path(output_path)
    frame_datetime = video_metadata["datetime"] + timedelta(seconds=timestamp_sec)
    exif_datetime = frame_datetime.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")
    subsecond = f"{frame_datetime.microsecond:06d}".encode("ascii")
    timestamp = format_timestamp(timestamp_sec)
    comment = (
        f"PCP v{VERSION} | Frame: {frame_idx} | Video time: {timestamp} | "
        f"Date source: {video_metadata['date_source']} | Source: {video_metadata['source']}"
    )

    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_datetime
    exif_dict["Exif"][piexif.ExifIFD.SubSecTime] = subsecond
    exif_dict["Exif"][piexif.ExifIFD.SubSecTimeOriginal] = subsecond
    exif_dict["Exif"][piexif.ExifIFD.SubSecTimeDigitized] = subsecond

    # UserComment records both the position in the source video and whether the
    # date came from embedded metadata or the file modification timestamp.
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
        b"UNICODE\x00" + comment.encode("utf-16be")
    )

    offset = _exif_offset(frame_datetime)
    if offset:
        offset_bytes = offset.encode("ascii")
        exif_dict["Exif"][piexif.ExifIFD.OffsetTime] = offset_bytes
        exif_dict["Exif"][piexif.ExifIFD.OffsetTimeOriginal] = offset_bytes
        exif_dict["Exif"][piexif.ExifIFD.OffsetTimeDigitized] = offset_bytes
    if video_metadata["make"]:
        exif_dict["0th"][piexif.ImageIFD.Make] = video_metadata["make"].encode("utf-8")
    if video_metadata["model"]:
        exif_dict["0th"][piexif.ImageIFD.Model] = video_metadata["model"].encode("utf-8")

    encoded_ok, encoded_frame = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not encoded_ok:
        raise OSError(f"could not encode JPEG: {output_path}")

    created = False
    try:
        # Exclusive creation prevents silent replacement of an earlier export.
        with output_path.open("xb") as output_file:
            created = True
            output_file.write(encoded_frame.tobytes())
        piexif.insert(piexif.dump(exif_dict), str(output_path))
    except Exception:
        if created:
            output_path.unlink(missing_ok=True)
        raise


def process_video(
    video_path,
    output_dir,
    model_path,
    strict_mode=False,
    smile_threshold=SMILE_THRESHOLD,
    eye_open_threshold=EYE_OPEN_THRESHOLD,
    blur_threshold=BLUR_THRESHOLD,
    samples_per_scene=FRAMES_PER_SCENE_CHECK,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"input video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {model_path}\nDownload it from: {MODEL_URL}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. READ METADATA AND DETECT SCENES
    print("Reading metadata and detecting scenes...")
    video_metadata = get_video_metadata(video_path)
    print(f"Photo date source: {video_metadata['date_source']}")

    video_stream = open_video(str(video_path))
    try:
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video_stream)

        # A video without cuts is still one valid scene.
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        scene_fps = video_stream.frame_rate
    finally:
        video_stream.capture.release()

    if not scene_list:
        print("No readable video frames found.")
        return 0, 0

    print(f"Found {len(scene_list)} scenes. Starting analysis...")

    # 2. INITIALIZE MEDIAPIPE FACE LANDMARKER
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        num_faces=MAX_FACES,
        output_face_blendshapes=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open input video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = scene_fps or 30.0

    saved_count = 0
    failed_count = 0

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as face_landmarker:
            # 3. ANALYZE EACH SCENE
            for scene_index, (start_time, end_time) in enumerate(scene_list, start=1):
                start_frame = start_time.frame_num
                end_frame = end_time.frame_num
                best_frame = None

                for frame_idx in sample_frame_indices(
                    start_frame, end_frame, samples_per_scene
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    frame_ok, frame = cap.read()
                    if not frame_ok:
                        continue

                    # Reject blur before running the more expensive ML model.
                    sharpness = frame_sharpness(frame)
                    if sharpness < blur_threshold:
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = face_landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    )
                    metrics = analyze_faces(
                        result.face_blendshapes,
                        smile_threshold,
                        eye_open_threshold,
                    )
                    if metrics is None:
                        continue

                    # Prefer the decoder presentation timestamp for variable
                    # frame-rate video, falling back to frame/fps when absent.
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if frame_idx and timestamp_ms <= 0:
                        timestamp_sec = frame_idx / fps
                    else:
                        timestamp_sec = max(0.0, timestamp_ms / 1000.0)

                    candidate = {
                        **metrics,
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "sharpness": sharpness,
                        "frame": frame.copy(),
                    }
                    if best_frame is None or candidate_rank(candidate) > candidate_rank(
                        best_frame
                    ):
                        best_frame = candidate

                # 4. MAKE THE FINAL DECISION FOR THIS SCENE
                if best_frame is None:
                    print(f"  Scene {scene_index}: Skipped (no sharp frame with a face)")
                    continue
                if strict_mode and not best_frame["perfect"]:
                    print(
                        f"  Scene {scene_index}: Skipped "
                        "(strict mode requires every face smiling with open eyes)"
                    )
                    continue

                timestamp = format_timestamp(best_frame["timestamp_sec"])
                quality_tag = "PERFECT" if best_frame["perfect"] else "FALLBACK"
                filename = (
                    f"PCP_Scene{scene_index:03d}_{quality_tag}_"
                    f"T{timestamp.replace(':', '-')}.jpg"
                )
                output_path = output_dir / filename
                if output_path.exists():
                    print(f"  Scene {scene_index}: Skipped (already exists: {filename})")
                    continue

                # 5. EXPORT JPEG AND EXIF
                try:
                    save_frame_with_exif(
                        best_frame["frame"],
                        output_path,
                        best_frame["frame_idx"],
                        best_frame["timestamp_sec"],
                        video_metadata,
                    )
                except Exception as exc:
                    failed_count += 1
                    print(f"  Scene {scene_index}: Failed to save ({exc})")
                    continue

                print(
                    f"  Scene {scene_index}: Saved ({quality_tag}) | "
                    f"Faces: {best_frame['face_count']} | "
                    f"Eyes: {best_frame['eye_open']:.2f} | "
                    f"Smile: {best_frame['smile']:.2f}"
                )
                saved_count += 1
    finally:
        cap.release()

    print(
        f"\nDone! Saved {saved_count} photos to '{output_dir}'"
        + (f"; {failed_count} failed" if failed_count else "")
    )
    return saved_count, failed_count


def build_parser():
    parser = argparse.ArgumentParser(
        description=f"PhotoCherryPick (PCP) v{VERSION} - Extract the best face frames."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("output", help="Destination folder for extracted photos")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).with_name(MODEL_FILENAME)),
        help=f"Face Landmarker model path (default: {MODEL_FILENAME} next to pcp.py)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Save only frames where every detected face smiles with open eyes",
    )
    parser.add_argument(
        "--smile-threshold", type=float, default=SMILE_THRESHOLD, metavar="0..1"
    )
    parser.add_argument(
        "--eye-open-threshold", type=float, default=EYE_OPEN_THRESHOLD, metavar="0..1"
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=BLUR_THRESHOLD, metavar="N"
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=FRAMES_PER_SCENE_CHECK,
        metavar="N",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 <= args.smile_threshold <= 1.0:
        parser.error("--smile-threshold must be between 0 and 1")
    if not 0.0 <= args.eye_open_threshold <= 1.0:
        parser.error("--eye-open-threshold must be between 0 and 1")
    if args.blur_threshold < 0:
        parser.error("--blur-threshold cannot be negative")
    if args.samples_per_scene < 1:
        parser.error("--samples-per-scene must be at least 1")

    try:
        _, failed_count = process_video(
            args.video,
            args.output,
            args.model,
            strict_mode=args.strict,
            smile_threshold=args.smile_threshold,
            eye_open_threshold=args.eye_open_threshold,
            blur_threshold=args.blur_threshold,
            samples_per_scene=args.samples_per_scene,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
