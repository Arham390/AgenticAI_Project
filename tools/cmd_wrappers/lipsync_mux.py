from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import imageio.v2 as iio


def _find_frames(path: Path) -> List[Path]:
    png = sorted(path.glob("frame_*.png"))
    if png:
        return png
    jpg = sorted(path.glob("frame_*.jpg"))
    if jpg:
        return jpg
    return sorted(path.glob("frame_*.jpeg"))


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def render_video(frames: List[Path], fps: int, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(out_mp4), fps=max(6, fps), codec="libx264")
    try:
        for fp in frames:
            writer.append_data(iio.imread(fp))
    finally:
        writer.close()


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 2048


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scene mp4 from frame sequence and mux audio.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    frame_dir = Path(args.frames)
    audio_path = Path(args.audio)
    output_path = Path(args.output)

    frames = _find_frames(frame_dir)
    if not frames:
        raise SystemExit("No frames found for lipsync mux")

    silent_path = Path(tempfile.gettempdir()) / f"{args.scene}_silent_mux.mp4"
    render_video(frames=frames, fps=max(6, int(args.fps)), out_mp4=silent_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists() and mux_audio(silent_path, audio_path, output_path):
        print(f"scene={args.scene} output={output_path}")
        return 0

    shutil.copyfile(silent_path, output_path)
    print(f"scene={args.scene} output={output_path} (video-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
