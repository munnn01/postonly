"""Standalone real-codec smoke test; no dataset, weights or proxy required."""
import argparse
import json
import subprocess

import numpy as np
import torch
from torch import nn


def run(command, data):
    result = subprocess.run(command, input=data, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    source = torch.rand(1, 4, 3, 32, 32)
    raw = source[0].permute(0, 2, 3, 1).mul(255).round().byte().numpy().tobytes()
    fmt = "h264" if args.codec == "h264" else "hevc"
    encoder = "libx264" if args.codec == "h264" else "libx265"
    command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "32x32", "-r", "30", "-i", "pipe:0",
               "-an", "-c:v", encoder, "-threads", "1", "-preset", "ultrafast",
               "-qp", "35", "-pix_fmt", "yuv420p"]
    if args.codec == "h265":
        command += ["-x265-params", "pools=none:frame-threads=1:log-level=error"]
    bitstream = run(command + ["-f", fmt, "pipe:1"], raw)
    decoded_bytes = run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
                         "-f", fmt, "-i", "pipe:0", "-threads", "1", "-f", "rawvideo",
                         "-pix_fmt", "rgb24", "pipe:1"], bitstream)
    assert len(decoded_bytes) == 4 * 32 * 32 * 3
    decoded = torch.from_numpy(np.frombuffer(decoded_bytes, dtype=np.uint8).copy())
    decoded = decoded.reshape(4, 32, 32, 3).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
    post = nn.Conv3d(3, 3, kernel_size=3, padding=1)
    nn.init.zeros_(post.weight)
    nn.init.zeros_(post.bias)
    def restore():
        return (decoded + post(decoded.transpose(1, 2)).transpose(1, 2)).clamp(0, 1)
    assert torch.equal(restore(), decoded)
    optimizer = torch.optim.SGD(post.parameters(), lr=0.1)
    before = float((decoded - source).square().mean())
    for _ in range(3):
        optimizer.zero_grad()
        loss = (restore() - source).square().mean()
        loss.backward()
        assert post.weight.grad is not None and torch.isfinite(post.weight.grad).all()
        optimizer.step()
    after = float((restore().detach() - source).square().mean())
    bpp = 8 * len(bitstream) / (4 * 32 * 32)
    assert bpp > 0 and post.weight.detach().abs().sum() > 0
    print(json.dumps({"synthetic_only": True, "codec": args.codec, "qp": 35,
                      "bitstream_bytes": len(bitstream), "anchor_bpp": bpp,
                      "postonly_bpp": bpp, "mse_before": before, "mse_after": after,
                      "identity_check": "passed", "gradient_check": "passed",
                      "note": "Three MSE optimization steps on synthetic input; not a Top-1 or BD-rate benchmark."}, indent=2))


if __name__ == "__main__":
    main()
