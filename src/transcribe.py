import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


MODEL_SIZE = "small"


def transcribe_video(video_path: Path) -> dict:
    # load the model
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )

    # transcribe the video and automatically detect the language
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
    )

    # convert whisper segments into a json-friendly structure
    result_segments = []

    for segment in segments:
        result_segments.append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
        )

    # combine all segment texts into one full transcript
    transcript = " ".join(
        segment["text"] for segment in result_segments
    )

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "transcript": transcript,
        "segments": result_segments,
    }


def parse_arguments():
    # define the command-line arguments
    parser = argparse.ArgumentParser(
        description="Transcribe a video using faster-whisper."
    )

    parser.add_argument(
        "video_file",
        type=Path,
        help="Path to the video file to transcribe.",
    )

    parser.add_argument(
        "--delete-input",
        action="store_true",
        help="Delete the input video after the transcript is saved.",
    )

    return parser.parse_args()


def main():
    # read the command-line arguments
    args = parse_arguments()
    video_path = args.video_file

    # stop if the input file does not exist
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return

    # create the output directory if it does not already exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # generate the output json filename from the input filename
    output_path = output_dir / f"{video_path.stem}.json"

    print(f"Transcribing: {video_path}")

    result = transcribe_video(video_path)

    # save the transcription result as json
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # delete the input file only if the user requested it
    if args.delete_input:
        video_path.unlink()
        print(f"Deleted input file: {video_path}")

    print(f"Detected language: {result['language']}")
    print(f"Transcript: {result['transcript']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()