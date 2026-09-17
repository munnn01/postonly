"""Create a diverse synthetic video pool under an ignored outputs/ folder.

Labels come from the frozen analyzer's argmax on the *evaluation-identical*
clip (same VideoDataset sampling), so anchor top1 is high at low QP and can
drop at high QP, giving an RD curve with multiple envelope points.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from postonly.core import FrozenAnalyzer, VideoDataset

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/diverse_videos")
staging = root / "_staging"
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)
frame_count, size = 8, 128
analyzer = FrozenAnalyzer("r3d_18").eval()
frames_opt = dict(frames=4, stride=1, size=128, train=False)

paths = []
for i in range(8):
    clip_frames = []
    for t in range(frame_count):
        if i % 4 == 0:
            rng = np.random.default_rng(100 + i)
            frame = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        elif i % 4 == 1:
            frame = cv2.circle(np.zeros((size, size, 3), np.uint8),
                               (32 + 8 * t, 64), 20, (40, 160, 200), -1)
        elif i % 4 == 2:
            grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8)[None, :], (size, 1))
            frame = np.stack([np.roll(grad, 10 * t, axis=1)] * 3, axis=-1)
        else:
            frame = cv2.rectangle(np.zeros((size, size, 3), np.uint8),
                                  (8 * t, 8 * t), (8 * t + 40, 8 * t + 40), (200, 60, 30), -1)
        clip_frames.append(frame)
    path = staging / f"fixture{i}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (size, size))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")
    for frame in clip_frames:
        writer.write(frame)
    writer.release()
    paths.append(path)

dataset = VideoDataset(staging, [{"path": p.name, "label": 0} for p in paths],
                       frames=4, stride=1, size=128, train=False)
for index in range(len(dataset)):
    clip, _, _ = dataset[index]
    with torch.no_grad():
        top = int(analyzer(clip.unsqueeze(0))[0].argmax())
    destination = root / analyzer.categories[top]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / paths[index].name
    if target.exists():
        raise RuntimeError(f"Refusing overwrite: {target}")
    shutil.move(str(paths[index]), str(target))
shutil.rmtree(staging)
print("Wrote", sum(1 for _ in root.rglob("*.avi")), "videos under", root)
