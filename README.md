# PhotoCherryPick (PCP)

**PhotoCherryPick 0.3** is a local Python tool that extracts the best frame
containing faces from each scene of a video.

It was built to turn holiday videos into a small set of printable photographs
without manually scrubbing through every recording.

## Features

- **Scene detection**: analyzes each shot independently; a video without cuts is
  treated as one scene.
- **Face expression analysis**: MediaPipe Face Landmarker evaluates smile and
  eye-blink blendshapes for every detected face.
- **Group-photo scoring**: the least successful face determines the frame
  quality, so one smiling person cannot hide somebody blinking.
- **Blur rejection**: skips frames below a configurable Laplacian sharpness
  threshold before running the ML model.
- **Faces only**: scenes without a detected face are intentionally skipped.
- **Strict and fallback modes**: default mode accepts the best neutral face
  frame; strict mode requires every detected face to smile with open eyes.
- **EXIF metadata**: writes `DateTime`, `DateTimeOriginal`,
  `DateTimeDigitized`, subsecond data, timezone offset, camera make/model when
  available, source frame, and video timestamp.
- **Explicit date fallback**: embedded capture metadata is preferred. If it is
  absent, PCP uses the source video's file modification time and records that
  choice in the EXIF `UserComment`.
- **Safe export**: existing JPEG files are never overwritten.
- **Local processing**: after dependencies and the model are downloaded, video
  frames are processed locally without cloud APIs.

## Requirements

- Python 3.9 or newer
- `curl`, a browser, or another downloader for the Face Landmarker model
- MediaInfo's native library if the `pymediainfo` wheel for the platform does
  not bundle it

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/francescodifilippo/PhotoCherryPick.git
   cd PhotoCherryPick
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:

   ```powershell
   py -m venv venv
   venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download the official MediaPipe Face Landmarker model next to `pcp.py`:

   ```bash
   curl -L --fail --output face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

   The model is intentionally not committed to this repository.

## Usage

Default mode saves the best sharp face frame in every scene. It prefers frames
where all detected faces smile with open eyes, then falls back to the best
neutral frame:

```bash
python pcp.py input_video.mp4 output
```

Strict mode skips every scene that lacks a frame where all detected faces smile
with open eyes:

```bash
python pcp.py input_video.mp4 output --strict
```

Useful calibration options:

```text
--smile-threshold 0..1
--eye-open-threshold 0..1
--blur-threshold N
--samples-per-scene N
--model PATH
```

Run `python pcp.py --help` for the complete command reference.

## Selection flow

1. Detect scene boundaries with PySceneDetect.
2. Sample at most 15 evenly spaced frames per scene by default.
3. Reject blurred frames.
4. Detect up to 10 faces and score smiles and open eyes.
5. Prefer a perfect frame, then the frame with more detected faces, the best
   worst-face score, and finally the greatest sharpness.
6. Save the JPEG with EXIF metadata. The video timestamp is taken from the
   decoder presentation timestamp when available, which also supports
   variable-frame-rate video.

Scenes without faces are reported as skipped and never exported.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers date parsing and fallback, group scoring, frame sampling,
timestamp rounding, best-frame ranking, EXIF output, and overwrite protection.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
