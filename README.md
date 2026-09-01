# PhotoCherryPick (PCP)

**PhotoCherryPick** is a lightweight, 100% local Python tool that automatically extracts the best frames from videos, scene by scene.

## Why I built this

I created this tool because I wanted to print a physical photo album of my holidays, but most of my best memories were captured in videos rather than photos. Manually scrubbing through hours of footage to find the perfect smiling faces was tedious, so I built PCP to automate the cherry-picking process. The goal was simple: get a set of high-quality stills, ready to be printed, without spending an evening pausing and rewinding the video player.

## Features

- **Scene Detection**: Analyzes the video scene by scene to ensure variety (no duplicate frames from the same shot).
- **Face Analysis**: Uses pre-trained ML models (Google MediaPipe) to detect faces, calculate Eye Aspect Ratio (EAR) for open eyes, and Mouth Aspect Ratio (MAR) for smiles.
- **Blur Rejection**: Automatically discards out-of-focus or motion-blurred frames before heavy processing.
- **EXIF Metadata Injection**: Reads video metadata (creation date, device model) and embeds it into the extracted photos, including the exact timestamp of the frame.
- **Strict / Fallback Modes**: Choose whether to save *only* perfect smiling frames, or fallback to the best available neutral frame if no smiles are detected.
- **100% Offline & Private**: No cloud APIs, no data leaving your machine, no model training required.

## Prerequisites

- Python 3.8 or higher
- [MediaInfo CLI](https://mediaarea.net/en/MediaInfo) (Optional but recommended for extracting video metadata. Usually auto-installed with `pymediainfo` on most systems).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/francescodifilippo/PhotoCherryPick.git
   cd PhotoCherryPick
    ```
2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Mode (Recommended)
Extracts the best frame per scene. Prefers smiles + open eyes, but will save a high-quality neutral frame if no smiles are found.
```bash
python pcp.py input_video.mp4 ./output_folder
```

### Strict Mode
Extracts a frame **ONLY** if it detects open eyes AND a smile. Scenes without smiling faces are skipped entirely.
```bash
python pcp.py input_video.mp4 ./output_folder --strict
```

## How it Works

1. **Scene Splitting**: Uses `PySceneDetect` to identify shot boundaries.
2. **Sampling**: Samples frames within each scene based on a configurable step.
3. **Scoring**: 
    - Checks for blur using Laplacian variance.
    - Detects facial landmarks via MediaPipe.
    - Calculates EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).
4. **Selection**: Picks the highest-scoring frame per scene based on the active mode.
5. **Export**: Saves the JPEG and uses `piexif` to inject the original video's creation date, device model, and exact frame timestamp into the EXIF `UserComment` field.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
