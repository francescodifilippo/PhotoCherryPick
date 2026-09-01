"""
PhotoCherryPick (PCP) v0.3
==============================
Extracts the best frames from videos, scene by scene.
Logic:
1. Detects scene changes.
2. For each scene, searches for faces.
3. Prefers smiles + open eyes.
4. --strict flag: saves ONLY if smile + open eyes are detected.
5. Default: saves the best available frame (even if neutral, as long as it's sharp).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import math
import piexif
import argparse
from datetime import datetime
from scenedetect import detect, ContentDetector, SceneManager, VideoStreamCv2

try:
    from pymediainfo import MediaInfo
    HAS_MEDIAINFO = True
except ImportError:
    HAS_MEDIAINFO = False

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.25       # Eyes open threshold
MAR_THRESHOLD = 0.45       # Smile threshold
BLUR_THRESHOLD = 100.0     # Sharpness threshold (Laplacian variance)
FRAMES_PER_SCENE_CHECK = 15 # Frames to sample per scene (balances speed/precision)

# MediaPipe Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    v1, v2 = calculate_distance(p2, p6), calculate_distance(p3, p5)
    h = calculate_distance(p1, p4)
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def calculate_mar(landmarks):
    p_top, p_bottom, p_left, p_right = [landmarks[i] for i in MOUTH]
    v, h = calculate_distance(p_top, p_bottom), calculate_distance(p_left, p_right)
    return v / h if h > 0 else 0.0

def is_blurry(frame, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def get_video_metadata(video_path):
    metadata = {
        "datetime": None, 
        "make": "Unknown", 
        "model": "Unknown", 
        "comment": f"Source: {os.path.basename(video_path)}"
    }
    if HAS_MEDIAINFO:
        try:
            for track in MediaInfo.parse(video_path).tracks:
                if track.track_type == "General" and hasattr(track, 'tagged_date') and track.tagged_date:
                    # Format date for EXIF (YYYY:MM:DD HH:MM:SS)
                    raw_date = track.tagged_date.replace("UTC ", "").replace(" ", "T")
                    metadata["datetime"] = raw_date[:19].replace("T", " ").replace("-", ":")
                elif track.track_type == "Video":
                    if hasattr(track, 'writing_library') and track.writing_library: 
                        metadata["model"] = track.writing_library
                    if hasattr(track, 'make') and track.make: 
                        metadata["make"] = track.make
        except Exception: 
            pass
    return metadata

def save_frame_with_exif(frame, output_path, frame_idx, timestamp_str, video_metadata):
    cv2.imwrite(output_path, frame)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    
    exif_dict["0th"][piexif.ImageIFD.DateTime] = video_metadata["datetime"] or datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif_dict["0th"][piexif.ImageIFD.Make] = video_metadata["make"]
    exif_dict["0th"][piexif.ImageIFD.Model] = video_metadata["model"]
    
    # UserComment: Custom field for timestamp and frame info
    custom_comment = f"PCP v0.3 | Frame: {frame_idx} | Time: {timestamp_str} | {video_metadata['comment']}"
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = b"UNICODE\x00\x00" + custom_comment.encode('utf-16be')
    
    try:
        piexif.insert(piexif.dump(exif_dict), output_path)
    except Exception: 
        pass

def format_timestamp(seconds, fps):
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}:{(seconds % 60):06.3f}"

def process_video(video_path, output_dir, strict_mode):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    print("Reading metadata and detecting scenes...")
    video_metadata = get_video_metadata(video_path)
    
    # 1. SCENE DETECTION
    video_stream = VideoStreamCv2(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    scene_manager.detect_scenes(video_stream)
    scene_list = scene_manager.get_scene_list()
    video_stream.release()

    if not scene_list:
        print("No scenes detected or video is too short.")
        return

    print(f"Found {len(scene_list)} scenes. Starting analysis...")

    # 2. INITIALIZE MEDIAPIPE
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved_count = 0

    # 3. SCENE-BY-SCENE ANALYSIS
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        scene_duration = end_frame - start_frame
        
        best_frame_data = None # Will hold: {'frame_idx': x, 'frame': img, 'score': y, 'type': 'perfect'|'fallback'}

        # Sample frames within the scene
        step = max(1, scene_duration // FRAMES_PER_SCENE_CHECK)
        for frame_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: 
                continue

            if is_blurry(frame): 
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Use the first detected face as reference
                landmarks = results.multi_face_landmarks[0].landmark
                avg_ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2.0
                mar = calculate_mar(landmarks)

                is_perfect = (avg_ear > EAR_THRESHOLD) and (mar > MAR_THRESHOLD)
                score = avg_ear + mar # Simple score: higher is better

                frame_type = 'perfect' if is_perfect else 'fallback'
                
                # Selection Logic: 
                # - If 'perfect', it overwrites anything.
                # - If 'fallback', keep it only if we don't have a 'perfect' frame for this scene yet.
                if is_perfect or (best_frame_data is None or best_frame_data['type'] == 'fallback'):
                    best_frame_data = {
                        'frame_idx': frame_idx,
                        'frame': frame.copy(),
                        'score': score,
                        'type': frame_type,
                        'ear': avg_ear,
                        'mar': mar
                    }

        # 4. FINAL DECISION FOR THE SCENE
        if best_frame_data:
            if strict_mode and best_frame_data['type'] == 'fallback':
                print(f"  Scene {i+1}: Skipped (strict mode, requires smile + open eyes)")
                continue

            timestamp_sec = best_frame_data['frame_idx'] / fps
            timestamp_str = format_timestamp(timestamp_sec, fps)
            quality_tag = "PERFECT" if best_frame_data['type'] == 'perfect' else "FALLBACK"
            
            filename = f"PCP_Scene{i+1:03d}_{quality_tag}_T{timestamp_str.replace(':', '-')}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            save_frame_with_exif(best_frame_data['frame'], output_path, best_frame_data['frame_idx'], timestamp_str, video_metadata)
            print(f"  Scene {i+1}: Saved ({quality_tag}) | EAR: {best_frame_data['ear']:.2f} | MAR: {best_frame_data['mar']:.2f}")
            saved_count += 1

    cap.release()
    face_mesh.close()
    print(f"\nDone! Saved {saved_count} photos to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhotoCherryPick (PCP) v3.0 - Extract the best frames from videos.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("output", help="Destination folder for extracted photos")
    parser.add_argument("--strict", action="store_true", help="Save ONLY frames with smiles and open eyes (discard fallbacks)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.video):
        process_video(args.video, args.output, strict_mode=args.strict)
    else:
        print(f"Error: The video file '{args.video}' does not exist.")
