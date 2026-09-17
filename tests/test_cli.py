"""Integration test exercises actual trainer/evaluator without downloading weights."""
import json
from argparse import Namespace

import numpy as np
import torch
from torch import nn

from postonly import cli


class TinyAnalyzer(nn.Module):
    categories = ["class_a", "class_b"]

    def __init__(self, name="unused"):
        super().__init__()
        self.head = nn.Linear(3, 2)
        self.requires_grad_(False)

    def forward(self, clip):
        return self.head(clip.mean(dim=(1, 3, 4)))


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples, **kwargs):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return torch.full((4, 3, 16, 16), 0.2 + 0.6 * sample["label"]), sample["label"], index


class TinyCodec:
    def __init__(self, codec="h264", qp=35, **kwargs):
        self.qp = qp

    def __call__(self, clips):
        return clips.clone(), torch.full((len(clips),), 1.0 / self.qp)


def test_training_and_checkpoint_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(cli, "FrozenAnalyzer", TinyAnalyzer)
    monkeypatch.setattr(cli, "VideoDataset", TinyDataset)
    monkeypatch.setattr(cli, "RealCodec", TinyCodec)
    monkeypatch.setattr(cli, "inventory", lambda *a: [
        {"path": f"class_{i % 2}/{i}.mp4", "label": i % 2} for i in range(12)
    ])
    args = Namespace(output=str(tmp_path / "train"), data_root=str(tmp_path), seed=42,
                     analyzer="tiny", device="cpu", limit_train=8, limit_val=4,
                     frames=4, stride=1, size=16, workers=0, batch_size=2,
                     channels=8, max_residual=0.1, lr=1e-3, weight_decay=0.0,
                     epochs=1, qps=[30, 35, 40, 45], codec="h264", ffmpeg="unused",
                     preset="medium", fps=30, mse_weight=0.1, bootstrap=0, command="train")
    cli.train(args)
    output = tmp_path / "train"
    assert (output / "best.pt").is_file()
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "data_root" not in checkpoint["config"]
    assert len(json.loads((output / "split_manifest.json").read_text())["val"]) == 4
    args.checkpoint = str(output / "best.pt")
    args.manifest = str(output / "split_manifest.json")
    args.output = str(tmp_path / "evaluate")
    cli.evaluate_checkpoint(args)
    assert (tmp_path / "evaluate" / "bd_rate.json").is_file()


def test_nonempty_output_refused(tmp_path):
    (tmp_path / "existing.txt").write_text("preserve")
    import pytest
    with pytest.raises(ValueError):
        cli.new_output(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "preserve"
