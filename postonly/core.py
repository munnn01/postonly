"""Small standalone modules for real-codec post-only experiments."""
from __future__ import annotations

import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from torch import nn


class PostProcessor(nn.Module):
    def __init__(self, channels=16, max_residual=0.1):
        super().__init__()
        if channels < 1 or not 0 < max_residual <= 1:
            raise ValueError("Invalid channels or residual range")
        self.features = nn.Sequential(nn.Conv3d(4, channels, 3, padding=1), nn.SiLU(),
                                      nn.Conv3d(channels, channels, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_residual = max_residual

    def forward(self, x, qp):
        video = x.transpose(1, 2)
        q = torch.full_like(video[:, :1], float(qp) / 51)
        residual = self.head(self.features(torch.cat((video, q), dim=1))).tanh()
        return (video + self.max_residual * residual).clamp(0, 1).transpose(1, 2)


class FrozenAnalyzer(nn.Module):
    def __init__(self, name="r3d_18"):
        super().__init__()
        from torchvision.models import video
        options = {"r3d_18": video.R3D_18_Weights, "mc3_18": video.MC3_18_Weights,
                   "r2plus1d_18": video.R2Plus1D_18_Weights}
        if name not in options:
            raise ValueError(f"Supported analyzers: {list(options)}")
        weights = options[name].DEFAULT
        self.net = getattr(video, name)(weights=weights)
        self.transform = weights.transforms()
        self.categories = list(weights.meta["categories"])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x):
        return self.net(self.transform(x))


class RealCodec:
    def __init__(self, codec="h264", qp=35, ffmpeg="ffmpeg", preset="medium", fps=30):
        if codec not in ("h264", "h265") or not 0 <= qp <= 51 or fps <= 0:
            raise ValueError("Invalid codec, QP or FPS")
        self.codec, self.qp, self.ffmpeg, self.preset, self.fps = codec, qp, ffmpeg, preset, fps

    def run(self, args, data):
        result = subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error"] + args,
                                input=data, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result.stdout

    def __call__(self, clips):
        if clips.ndim != 5 or clips.shape[2] != 3 or min(clips.shape) < 1:
            raise ValueError("Expected nonempty BTCHW RGB video")
        if not torch.isfinite(clips).all() or clips.min() < 0 or clips.max() > 1:
            raise ValueError("Video must be finite in [0,1]")
        _, frames, _, height, width = clips.shape
        if height % 2 or width % 2:
            raise ValueError("yuv420p requires even spatial dimensions")
        fmt = "h264" if self.codec == "h264" else "hevc"
        encoder = "libx264" if self.codec == "h264" else "libx265"
        videos, rates = [], []
        for clip in clips.detach().cpu():
            raw = clip.permute(0, 2, 3, 1).mul(255).round().byte().numpy().tobytes()
            command = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                       "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", encoder,
                       "-threads", "1", "-preset", self.preset, "-qp", str(self.qp),
                       "-pix_fmt", "yuv420p"]
            if self.codec == "h265":
                command += ["-x265-params", "pools=none:frame-threads=1:log-level=error"]
            stream = self.run(command + ["-f", fmt, "pipe:1"], raw)
            decoded = self.run(["-threads", "1", "-f", fmt, "-i", "pipe:0", "-threads", "1",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stream)
            if len(decoded) != frames * height * width * 3:
                raise RuntimeError("Decoded frame count or dimensions differ")
            array = np.frombuffer(decoded, dtype=np.uint8).copy().reshape(frames, height, width, 3)
            videos.append(torch.from_numpy(array).permute(0, 3, 1, 2).float().div(255))
            rates.append(len(stream) * 8 / (frames * height * width))
        return torch.stack(videos).to(clips), torch.tensor(rates, dtype=torch.float64)


def inventory(root, categories):
    root = Path(root).resolve()
    mapping = {name: i for i, name in enumerate(categories)}
    samples = []
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        videos = sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions)
        if videos and directory.name not in mapping:
            raise ValueError(f"Unknown class directory: {directory.name}")
        for path in videos:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Video path escapes data root")
            samples.append({"path": path.relative_to(root).as_posix(), "label": mapping[directory.name]})
    if not samples:
        raise ValueError("No labeled videos found")
    return samples


def split_samples(samples, train_count, val_count, seed):
    if min(train_count, val_count) < 1 or len(samples) < train_count + val_count:
        raise ValueError("Not enough videos for requested exact train/val counts")
    paths = [sample["path"] for sample in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate sample paths")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["label"]].append(dict(sample))
    for group in groups.values():
        rng.shuffle(group)
    def take(count, reserve):
        selected = []
        while len(selected) < count:
            labels = [label for label, group in groups.items() if len(group) > reserve]
            if not labels:
                raise ValueError("Cannot reserve training members of each validation class")
            rng.shuffle(labels)
            for label in labels:
                if len(selected) == count:
                    break
                selected.append(groups[label].pop())
        return selected
    val = take(val_count, 1)
    train = take(train_count, 0)
    return train, val


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples, frames=16, stride=2, size=128, train=False):
        self.root, self.samples = Path(root).resolve(), samples
        self.frames, self.stride, self.size, self.train = frames, stride, size, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = (self.root / sample["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Sample path escapes root")
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"Unreadable video: {sample['path']}")
        span = (self.frames - 1) * self.stride + 1
        last_start = max(0, len(frames) - span)
        start = random.randint(0, last_start) if self.train else last_start // 2
        selected = [frames[min(start + i * self.stride, len(frames) - 1)] for i in range(self.frames)]
        height, width = selected[0].shape[:2]
        short = int(self.size * 1.125)
        scale = short / min(height, width)
        h, w = max(self.size, round(height * scale)), max(self.size, round(width * scale))
        top = random.randint(0, h-self.size) if self.train else (h-self.size)//2
        left = random.randint(0, w-self.size) if self.train else (w-self.size)//2
        array = np.stack([cv2.resize(f, (w, h))[top:top+self.size, left:left+self.size] for f in selected])
        if self.train and random.random() < 0.5:
            array = array[:, :, ::-1]
        clip = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div(255)
        return clip, int(sample["label"]), index


def bd_rate(rows):
    if len({r.get("codec", "unspecified") for r in rows}) > 1:
        raise ValueError("Calculate BD-rate separately for each codec")
    groups = defaultdict(list)
    for row in rows:
        if not math.isfinite(float(row["bpp"])) or row["bpp"] <= 0 or not 0 <= row["top1"] <= 1:
            raise ValueError("Invalid rate or accuracy")
        groups[(row["method"], row["qp"])].append(row)
    curves = {}
    for method in ("anchor", "postonly"):
        points = sorted((float(np.mean([r["bpp"] for r in group])),
                         100 * float(np.mean([r["top1"] for r in group])))
                        for (m, _), group in groups.items() if m == method)
        best_at_rate = {}
        for rate, quality in points:
            best_at_rate[rate] = max(quality, best_at_rate.get(rate, -math.inf))
        kept = []
        for rate, quality in sorted(best_at_rate.items()):
            if not kept or quality > kept[-1][1]:
                kept.append((rate, quality))
        curves[method] = kept
    result = {"bd_rate_percent": None, "quality_min": None, "quality_max": None,
              "interpolation": "pchip", "curves": curves}
    a, p = curves["anchor"], curves["postonly"]
    if min(len(a), len(p)) < 2:
        return result
    lo, hi = max(a[0][1], p[0][1]), min(a[-1][1], p[-1][1])
    result.update(quality_min=lo, quality_max=hi)
    if hi <= lo:
        return result
    fa = PchipInterpolator([q for r, q in a], np.log([r for r, q in a]))
    fp = PchipInterpolator([q for r, q in p], np.log([r for r, q in p]))
    result["bd_rate_percent"] = float(100 * np.expm1((fp.integrate(lo, hi)-fa.integrate(lo, hi))/(hi-lo)))
    return result


def paired_bootstrap(rows, samples=1000, seed=2026):
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    if not groups:
        raise ValueError("No paired samples")
    expected = {(r["method"], r["qp"]) for r in rows}
    qps = {r["qp"] for r in rows}
    if expected != {(m, q) for m in ("anchor", "postonly") for q in qps}:
        raise ValueError("Missing anchor/postonly QP pair")
    for group in groups.values():
        if len(group) != len(expected) or {(r["method"], r["qp"]) for r in group} != expected:
            raise ValueError("Incomplete or duplicate paired video rows")
    ids = list(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = rng.integers(len(ids), size=len(ids))
        resampled = [r for i in selected for r in groups[ids[i]]]
        value = bd_rate(resampled)["bd_rate_percent"]
        if value is not None and math.isfinite(value):
            values.append(value)
    interval = [float(v) for v in np.percentile(values, [2.5, 97.5])] if values else None
    return {"confidence": 0.95, "interval_percent": interval, "valid": len(values),
            "invalid": samples-len(values), "samples": samples, "seed": seed}
