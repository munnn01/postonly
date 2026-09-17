"""Train and evaluate a post-only recognition baseline with a real codec."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .core import (
    FrozenAnalyzer, PostProcessor, RealCodec, VideoDataset,
    bd_rate, inventory, paired_bootstrap, split_samples,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def codec_for(args, qp):
    return RealCodec(codec=args.codec, qp=qp, ffmpeg=args.ffmpeg, preset=args.preset, fps=args.fps)


def make_loader(root, samples, args, train):
    dataset = VideoDataset(root, samples, frames=args.frames, stride=args.stride,
                           size=args.size, train=train)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, drop_last=False,
                      generator=torch.Generator().manual_seed(args.seed),
                      worker_init_fn=worker_seed)


@torch.no_grad()
def evaluate(model, analyzer, loader, args):
    model.eval()
    analyzer.eval()
    rows = []
    for qp in args.qps:
        codec = codec_for(args, qp)
        for clips, labels, indices in loader:
            decoded, rates = codec(clips)
            restored = model(decoded.to(args.device), qp)
            for method, video in (("anchor", decoded.to(args.device)), ("postonly", restored)):
                logits = analyzer(video)
                correct = logits.argmax(1).cpu().eq(labels)
                mse = (video.cpu() - clips).square().flatten(1).mean(1)
                for index, rate, hit, error in zip(indices, rates, correct, mse):
                    rows.append({"sample_id": str(int(index)), "codec": args.codec,
                                 "qp": qp, "method": method, "bpp": float(rate),
                                 "top1": int(hit), "mse": float(error)})
    return rows


def write_results(output, rows, bootstrap):
    with (output / "per_video_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = bd_rate(rows)
    if bootstrap:
        result["bootstrap"] = paired_bootstrap(rows, samples=bootstrap, seed=2026)
    result["scope"] = "postonly vs anchor; same bitstream at each sample/QP; short-clip encoding"
    save_json(output / "bd_rate.json", result)
    return result


def train(args):
    seed_all(args.seed)
    output = new_output(args.output)
    analyzer = FrozenAnalyzer(args.analyzer).to(args.device).eval()
    samples = inventory(args.data_root, analyzer.categories)
    train_samples, val_samples = split_samples(samples, args.limit_train, args.limit_val, args.seed)
    manifest = {"seed": args.seed, "categories": analyzer.categories,
                "train": train_samples, "val": val_samples}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    save_json(output / "split_manifest.json", manifest)
    train_loader = make_loader(args.data_root, train_samples, args, True)
    val_loader = make_loader(args.data_root, val_samples, args, False)
    model = PostProcessor(channels=args.channels, max_residual=args.max_residual).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    qp_rng = random.Random(args.seed + 1)
    best_score = float("inf")
    valid_best = False
    codecs = {q: codec_for(args, q) for q in args.qps}
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for clips, labels, _ in train_loader:
            qp = qp_rng.choice(args.qps)
            with torch.no_grad():
                decoded, _ = codecs[qp](clips)
            clips = clips.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            restored = model(decoded.to(args.device), qp)
            # Frozen analyzer weights still permit gradients with respect to restored video.
            ce = nn.functional.cross_entropy(analyzer(restored), labels)
            loss = ce + args.mse_weight * nn.functional.mse_loss(restored, clips)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
        rows = evaluate(model, analyzer, val_loader, args)
        details = bd_rate(rows)
        value = details["bd_rate_percent"]
        accuracy = float(np.mean([r["top1"] for r in rows if r["method"] == "postonly"]))
        score = value if value is not None else -accuracy
        selected = (value is not None and not valid_best) or ((value is not None) == valid_best and score < best_score)
        payload = {"format_version": 1, "model": model.state_dict(), "epoch": epoch + 1,
                   "config": {k: v for k, v in vars(args).items() if k not in ("data_root", "output", "ffmpeg")},
                   "split_sha256": digest, "categories": analyzer.categories,
                   "validation_bd_rate": value}
        torch.save(payload, output / "last.pt")
        if selected:
            torch.save(payload, output / "best.pt")
            best_score, valid_best = score, value is not None
        history.append({"epoch": epoch + 1, "loss": loss_sum / count,
                        "val_top1_percent": accuracy * 100, "bd_rate_percent": value,
                        "selected": selected})
        save_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    final_rows = evaluate(model, analyzer, val_loader, args)
    print(json.dumps(write_results(output, final_rows, args.bootstrap), indent=2))


def evaluate_checkpoint(args):
    seed_all(args.seed)
    output = new_output(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = checkpoint["config"]
    for field in ("codec", "analyzer", "channels", "max_residual", "frames", "stride", "size", "fps", "preset", "qps"):
        setattr(args, field, config[field])
    analyzer = FrozenAnalyzer(args.analyzer).to(args.device).eval()
    if list(analyzer.categories) != checkpoint["categories"]:
        raise ValueError("Analyzer categories differ from checkpoint")
    model = PostProcessor(channels=args.channels, max_residual=args.max_residual).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if digest != checkpoint["split_sha256"]:
            raise ValueError("Validation manifest differs from training checkpoint")
        samples = manifest["val"]
        split = "validation-used-for-selection"
    else:
        samples = inventory(args.data_root, analyzer.categories)
        split = "user-supplied-test-root; independence must be verified by user"
    if not samples:
        raise ValueError("No evaluation videos")
    save_json(output / "evaluation_protocol.json", {"split": split, "videos": len(samples), "config": config})
    rows = evaluate(model, analyzer, make_loader(args.data_root, samples, args, False), args)
    print(json.dumps(write_results(output, rows, args.bootstrap), indent=2))


def smoke(args):
    seed_all(7)
    output = new_output(args.output)
    source = torch.rand(1, 4, 3, 32, 32)
    model = PostProcessor(channels=8)
    # Synthetic classifier tests differentiation only; it is not a recognition benchmark.
    analyzer = nn.Sequential(nn.Flatten(), nn.Linear(source.numel(), 3)).eval()
    analyzer.requires_grad_(False)
    records = []
    for codec_name in ("h264", "h265"):
        decoded, rates = RealCodec(codec_name, 35, ffmpeg=args.ffmpeg, preset="ultrafast")(source)
        with torch.no_grad():
            identity = model(decoded, 35)
        if not torch.equal(identity, decoded):
            raise AssertionError("Fresh postprocessor must be exactly identity")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad()
        restored = model(decoded, 35)
        loss = nn.functional.cross_entropy(analyzer(restored), torch.tensor([1]))
        loss.backward()
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert all(p.grad is None for p in analyzer.parameters())
        records.append({"codec": codec_name, "qp": 35, "anchor_bpp": float(rates[0]),
                        "postonly_bpp": float(rates[0]), "decoded_mse": float((decoded-source).square().mean()),
                        "identity_max_error": float((identity-decoded).abs().max()),
                        "gradient_test": "passed"})
        model = PostProcessor(channels=8)
    save_json(output / "smoke.json", {"synthetic_only": True, "results": records})
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "smoke"))
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codec", choices=("h264", "h265"), default="h264")
    parser.add_argument("--qps", type=int, nargs="+", default=[30, 35, 40, 45])
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--analyzer", default="r3d_18")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--max-residual", type=float, default=0.1)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=2800)
    parser.add_argument("--limit-val", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.command != "smoke" and not args.data_root:
        parser.error("--data-root is required for train/evaluate")
    if args.command == "evaluate" and not args.checkpoint:
        parser.error("--checkpoint is required for evaluate")
    for name in ("epochs", "batch_size", "limit_train", "limit_val", "frames", "stride", "size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bootstrap < 0 or len(set(args.qps)) != len(args.qps) or any(q < 0 or q > 51 for q in args.qps):
        parser.error("Invalid bootstrap count or QP list")
    {"train": train, "evaluate": evaluate_checkpoint, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
