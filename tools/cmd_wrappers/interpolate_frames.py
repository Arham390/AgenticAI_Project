from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def _find_frames(path: Path) -> List[Path]:
    png = sorted(path.glob("frame_*.png"))
    if png:
        return png
    jpg = sorted(path.glob("frame_*.jpg"))
    if jpg:
        return jpg
    return sorted(path.glob("frame_*.jpeg"))


def interpolate_frames(input_dir: Path, output_dir: Path, factor: int) -> int:
    frames = _find_frames(input_dir)
    if len(frames) < 2:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_index = 1

    for idx in range(len(frames) - 1):
        a = Image.open(frames[idx]).convert("RGB")
        b = Image.open(frames[idx + 1]).convert("RGB")
        if a.size != b.size:
            b = b.resize(a.size, Image.Resampling.BICUBIC)

        out_path = output_dir / f"frame_{out_index:04d}.png"
        a.save(out_path)
        out_index += 1

        for step in range(1, factor):
            alpha = float(step) / float(factor)
            blend = Image.blend(a, b, alpha)
            out_path = output_dir / f"frame_{out_index:04d}.png"
            blend.save(out_path)
            out_index += 1

    tail = Image.open(frames[-1]).convert("RGB")
    out_path = output_dir / f"frame_{out_index:04d}.png"
    tail.save(out_path)
    return out_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpolate scene frame sequence into normalized frame_XXXX.png output.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()

    produced = interpolate_frames(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        factor=max(1, int(args.factor)),
    )
    if produced <= 0:
        raise SystemExit("No frames found to interpolate")

    print(f"scene={args.scene} interpolated_frames={produced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
