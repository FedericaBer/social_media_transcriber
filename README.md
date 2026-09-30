# Social Media Video Transcriber

A small Python project that transcribes video files using faster-whisper.

It automatically detects the spoken language and saves the transcript as JSON with timestamps.

## Setup

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Usage

Put a video inside the input folder, for example:

input/video.mp4

Run:

python src/transcribe.py input/video.mp4

The transcript will be saved as:

output/video.json

If you want the input video to be deleted after transcription:

python src/transcribe.py input/video.mp4 --delete-input

## Output example

{
  "language": "de",
  "language_probability": 0.99,
  "transcript": "Example transcript...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Example transcript..."
    }
  ]
}