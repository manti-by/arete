import torch

from audio_upscaler.losses import CombinedAudioLoss

BATCH = 2
SAMPLES = 16000


class TestCombinedAudioLoss:
    def test_loss_returns_scalar(self) -> None:
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        pred = torch.randn(BATCH, 1, SAMPLES)
        target = torch.randn(BATCH, 1, SAMPLES)
        total, _ = loss_fn(pred, target)
        assert total.ndim == 0, "Total loss should be a scalar"
        assert total.isfinite(), "Loss should be finite"

    def test_loss_components_present(self) -> None:
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        pred = torch.randn(BATCH, 1, SAMPLES)
        target = torch.randn(BATCH, 1, SAMPLES)
        _, components = loss_fn(pred, target)
        for key in ("l_time", "l_mr_stft", "l_mel", "l_highband", "total"):
            assert key in components, f"Missing component: {key}"
        for v in components.values():
            assert v is not None and v != float("nan"), f"Component {key} is nan"

    def test_perfect_prediction_lower_loss(self) -> None:
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        target = torch.randn(BATCH, 1, SAMPLES)
        bad_pred = torch.randn(BATCH, 1, SAMPLES)

        perfect_loss, _ = loss_fn(target.clone(), target)
        bad_loss, _ = loss_fn(bad_pred, target)

        assert perfect_loss.isfinite(), "Perfect loss should not be nan"
        assert bad_loss.isfinite(), "Bad loss should not be nan"
        assert perfect_loss.item() < bad_loss.item(), (
            "Loss on perfect prediction should be lower than on random noise"
        )

    def test_loss_backward(self) -> None:
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        pred = torch.randn(BATCH, 1, SAMPLES, requires_grad=True)
        target = torch.randn(BATCH, 1, SAMPLES)
        total, _ = loss_fn(pred, target)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_loss_l1_only_if_mel_loss_none(self) -> None:
        """When mel_loss is None (fallback mode), l_mel should equal l_mr."""
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        loss_fn.mel_loss = None
        pred = torch.randn(BATCH, 1, SAMPLES)
        target = torch.randn(BATCH, 1, SAMPLES)
        _, components = loss_fn(pred, target)
        assert components["l_mel"] == components["l_mr_stft"]

    def test_loss_different_sample_rates(self) -> None:
        for sr in [8000, 16000, 44100]:
            loss_fn = CombinedAudioLoss(sample_rate=sr)
            pred = torch.randn(BATCH, 1, SAMPLES)
            target = torch.randn(BATCH, 1, SAMPLES)
            total, _ = loss_fn(pred, target)
            assert total.isfinite(), f"Loss not finite at sample_rate={sr}"

    def test_loss_fallback_mrstft(self) -> None:
        loss_fn = CombinedAudioLoss(sample_rate=16000)
        loss_fn.mr_stft_loss = type(loss_fn.mr_stft_loss)([512], [128], [512])
        pred = torch.randn(BATCH, 1, SAMPLES)
        target = torch.randn(BATCH, 1, SAMPLES)
        total, _ = loss_fn(pred, target)
        assert total.isfinite()
