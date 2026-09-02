import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import piexif

import pcp


def blendshapes(**scores):
    return [
        SimpleNamespace(category_name=name, score=score)
        for name, score in scores.items()
    ]


class PhotoCherryPickTest(unittest.TestCase):
    def test_parse_mediainfo_utc_dates(self):
        for value in ("UTC 2024-05-06 07:08:09", "2024-05-06 07:08:09 UTC"):
            parsed = pcp.parse_mediainfo_datetime(value)
            self.assertEqual(parsed, datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc))

    def test_metadata_prefers_embedded_capture_date(self):
        track = SimpleNamespace(
            track_type="General",
            recorded_date="UTC 2024-05-06 07:08:09",
            make="Example Make",
            model="Example Model",
        )
        with patch.object(
            pcp.MediaInfo, "parse", return_value=SimpleNamespace(tracks=[track])
        ):
            metadata = pcp.get_video_metadata("video.mp4")

        self.assertEqual(
            metadata["datetime"],
            datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc),
        )
        self.assertEqual(metadata["date_source"], "embedded video metadata (recorded_date)")
        self.assertEqual(metadata["make"], "Example Make")
        self.assertEqual(metadata["model"], "Example Model")

    def test_metadata_falls_back_to_video_file_mtime(self):
        timestamp = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            os.utime(video_path, (timestamp, timestamp))
            with patch.object(
                pcp.MediaInfo, "parse", return_value=SimpleNamespace(tracks=[])
            ):
                metadata = pcp.get_video_metadata(video_path)

        self.assertEqual(metadata["datetime"].timestamp(), timestamp)
        self.assertEqual(
            metadata["date_source"],
            "video file modification time (embedded original date unavailable)",
        )

    def test_samples_are_limited_and_avoid_scene_edges(self):
        indices = pcp.sample_frame_indices(0, 100, 15)
        self.assertEqual(len(indices), 15)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreater(indices[0], 0)
        self.assertLess(indices[-1], 99)

    def test_timestamp_rounding_carries_into_next_minute(self):
        self.assertEqual(pcp.format_timestamp(59.9996), "00:01:00.000")

    def test_face_metrics_use_the_worst_detected_face(self):
        happy = blendshapes(
            eyeBlinkLeft=0.1,
            eyeBlinkRight=0.2,
            mouthSmileLeft=0.9,
            mouthSmileRight=0.8,
        )
        neutral = blendshapes(
            eyeBlinkLeft=0.1,
            eyeBlinkRight=0.1,
            mouthSmileLeft=0.2,
            mouthSmileRight=0.2,
        )
        metrics = pcp.analyze_faces([happy, neutral], 0.5, 0.5)

        self.assertEqual(metrics["face_count"], 2)
        self.assertAlmostEqual(metrics["smile"], 0.2)
        self.assertFalse(metrics["perfect"])

    def test_candidate_rank_uses_score_instead_of_last_frame(self):
        better = {"perfect": True, "face_count": 1, "score": 0.9, "sharpness": 100}
        later = {"perfect": True, "face_count": 1, "score": 0.4, "sharpness": 500}
        self.assertGreater(pcp.candidate_rank(better), pcp.candidate_rank(later))

    def test_jpeg_has_original_date_source_and_is_not_overwritten(self):
        frame = np.full((16, 16, 3), 255, dtype=np.uint8)
        metadata = {
            "datetime": datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "date_source": (
                "video file modification time (embedded original date unavailable)"
            ),
            "make": None,
            "model": None,
            "source": "video.mp4",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "frame.jpg"
            pcp.save_frame_with_exif(frame, output_path, 10, 1.25, metadata)
            exif = piexif.load(str(output_path))
            comment = exif["Exif"][piexif.ExifIFD.UserComment]

            self.assertEqual(
                exif["Exif"][piexif.ExifIFD.DateTimeOriginal],
                b"2020:01:02 03:04:06",
            )
            self.assertIn("video file modification time", comment[8:].decode("utf-16be"))
            with self.assertRaises(FileExistsError):
                pcp.save_frame_with_exif(frame, output_path, 10, 1.25, metadata)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
