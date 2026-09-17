import pytest
import torch
from postonly.core import PostProcessor, bd_rate, paired_bootstrap, split_samples


def test_identity_and_gradients():
    torch.set_num_threads(1)
    model = PostProcessor(8)
    x = torch.rand(2, 4, 3, 16, 16)
    assert torch.equal(model(x, 35), x)
    frozen = torch.nn.Linear(3, 2).requires_grad_(False)
    logits = frozen(model(x, 35).mean(dim=(1, 3, 4)))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in frozen.parameters())


def curves(scale=1):
    return [{"sample_id": "0", "method": method, "qp": i, "bpp": rate * (scale if method == "postonly" else 1), "top1": quality}
            for method in ("anchor", "postonly")
            for i, (rate, quality) in enumerate([(0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.8, 0.8)])]


def test_bd_invariants():
    assert bd_rate(curves())["bd_rate_percent"] == pytest.approx(0, abs=1e-10)
    assert bd_rate(curves(0.9))["bd_rate_percent"] == pytest.approx(-10, abs=1e-10)
    rows = curves()
    for row in rows:
        row["top1"] = 0.5
    assert bd_rate(rows)["bd_rate_percent"] is None


def test_bootstrap_pairs():
    result = paired_bootstrap(curves(), samples=10)
    assert result["valid"] == 10
    assert result["interval_percent"] == pytest.approx([0, 0])
    with pytest.raises(ValueError):
        paired_bootstrap(curves()[:-1], samples=10)


def test_exact_split():
    samples = [{"path": f"{i%4}/{i}.mp4", "label": i%4} for i in range(40)]
    train, val = split_samples(samples, 28, 7, 42)
    assert len(train) == 28 and len(val) == 7
    assert not {r["path"] for r in train} & {r["path"] for r in val}
    assert (train, val) == split_samples(samples, 28, 7, 42)
    with pytest.raises(ValueError):
        split_samples(samples, 35, 7, 42)
