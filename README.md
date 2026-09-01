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
