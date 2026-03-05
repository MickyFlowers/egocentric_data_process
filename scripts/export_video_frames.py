#!/usr/bin/env python3
"""将视频逐帧导出为带帧号标号的图片。"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export video frames as images with frame index.")
    parser.add_argument("video_path", type=Path, help="Input video path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same name as video under <video_parent>/../<video_stem>_frames)",
    )
    parser.add_argument("--prefix", type=str, default="frame", help="Output filename prefix")
    parser.add_argument("--digits", type=int, default=4, help="Frame number zero-padding digits")
    parser.add_argument("--no-draw", action="store_true", help="Do not draw frame number on image")
    args = parser.parse_args()

    video_path = args.video_path.resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        # 默认输出到视频同目录下的 <stem>_frames
        out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    reader = imageio.get_reader(video_path)
    fmt = f"{args.prefix}_{{:0{args.digits}d}}.png"
    count = 0
    for frame_index, frame in enumerate(reader):
        out_path = out_dir / fmt.format(frame_index)
        if args.no_draw:
            imageio.imwrite(out_path, frame)
        elif has_cv2:
            img = np.asarray(frame)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 先画黑边再画白字，保证在任意背景上可读
            img = cv2.putText(
                img, str(frame_index), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA,
            )
            img = cv2.putText(
                img, str(frame_index), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA,
            )
            imageio.imwrite(out_path, img)
        else:
            imageio.imwrite(out_path, frame)
        count += 1
    reader.close()
    print(f"Exported {count} frames to {out_dir}")


if __name__ == "__main__":
    main()
