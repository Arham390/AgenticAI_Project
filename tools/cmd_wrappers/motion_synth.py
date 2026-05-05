from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def _find_keyframes(path: Path) -> List[Path]:
    png = sorted(path.glob("*_kf_*.png"))
    if png:
        return png
    jpg = sorted(path.glob("*_kf_*.jpg"))
    if jpg:
        return jpg
    return sorted(path.glob("*_kf_*.jpeg"))


def synthesize_motion(keyframe_dir: Path, output_dir: Path, fps: int) -> int:
    keyframes = _find_keyframes(keyframe_dir)
    if len(keyframes) < 2:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_index = 1
    steps_per_segment = max(4, min(12, max(1, int(fps)) // 2))

    for idx in range(len(keyframes) - 1):
        a = Image.open(keyframes[idx]).convert("RGB")
        b = Image.open(keyframes[idx + 1]).convert("RGB")
        if a.size != b.size:
            b = b.resize(a.size, Image.Resampling.BICUBIC)

        for step in range(steps_per_segment):
            alpha = float(step) / float(max(1, steps_per_segment - 1))
            frame = Image.blend(a, b, alpha)
            out_path = output_dir / f"frame_{out_index:04d}.png"
            frame.save(out_path)
            out_index += 1

    return out_index - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight motion synthesis from storyboard keyframes.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    produced = synthesize_motion(
        keyframe_dir=Path(args.keyframes),
        output_dir=Path(args.output),
        fps=max(1, int(args.fps)),
    )
    if produced <= 0:
        raise SystemExit("No keyframes found for motion synthesis")

    print(f"scene={args.scene} motion_frames={produced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
