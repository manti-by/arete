import torch

from arete.models import EMA, STFTUNet, WaveformUNet


BATCH = 2
CHANNELS = 1
SAMPLES = 44100


def make_dummy(batch: int = BATCH, channels: int = CHANNELS, samples: int = SAMPLES) -> torch.Tensor:
    return torch.randn(batch, channels, samples)


class TestWaveformUNet:
    def test_output_shape(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=16, depth=3)
        x = make_dummy()
        y = model(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_residual_not_zero(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=16, depth=3)
        x = make_dummy()
        with torch.no_grad():
            y = model(x)
        assert not torch.allclose(y, x), "Model should not be identity at init"

    def test_stereo(self) -> None:
        model = WaveformUNet(in_channels=2, base_channels=16, depth=3)
        x = make_dummy(channels=2)
        y = model(x)
        assert y.shape == x.shape

    def test_gradient_flows(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        x = make_dummy(batch=1, samples=8192)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(n > 0 for n in grad_norms), "Some gradients are zero"

    def test_different_depths(self) -> None:
        for depth in [2, 3, 4]:
            model = WaveformUNet(in_channels=1, base_channels=8, depth=depth)
            x = make_dummy(samples=16384)
            y = model(x)
            assert y.shape == x.shape, f"Failed at depth={depth}"

    def test_different_kernel_sizes(self) -> None:
        for ks in [5, 15, 31]:
            model = WaveformUNet(in_channels=1, base_channels=8, depth=2, kernel_size=ks)
            x = make_dummy(samples=8192)
            y = model(x)
            assert y.shape == x.shape, f"Failed at kernel_size={ks}"

    def test_very_small_input(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=4, depth=2)
        x = make_dummy(batch=1, samples=256)
        y = model(x)
        assert y.shape == x.shape

    def test_empty_batch(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        x = torch.randn(0, 1, 8192)
        y = model(x)
        assert y.shape == x.shape


class TestSTFTUNet:
    def test_output_shape(self) -> None:
        model = STFTUNet(n_fft=512, hop_length=128, sample_rate=44100, base_ch=8, depth=2)
        x = make_dummy(samples=8192)
        y = model(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_different_depths(self) -> None:
        for depth in [2, 3]:
            model = STFTUNet(n_fft=512, hop_length=128, sample_rate=44100, base_ch=8, depth=depth)
            x = make_dummy(samples=8192)
            y = model(x)
            assert y.shape == x.shape, f"Failed at depth={depth}"

    def test_gradient_flows(self) -> None:
        model = STFTUNet(n_fft=512, hop_length=128, sample_rate=44100, base_ch=8, depth=2)
        x = make_dummy(batch=1, samples=8192)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(n > 0 for n in grad_norms), "Some gradients are zero"

    def test_output_not_identical(self) -> None:
        model = STFTUNet(n_fft=512, hop_length=128, sample_rate=44100, base_ch=8, depth=2)
        x = make_dummy(samples=8192)
        with torch.no_grad():
            y = model(x)
        assert not torch.allclose(y, x, atol=1e-4), "STFT model should not be identity"


class TestEMA:
    def test_shadow_updates(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        ema = EMA(model, decay=0.9)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        ema.update(model)

        for s, p in zip(ema.shadow.parameters(), model.parameters(), strict=True):
            assert not torch.allclose(s, p), "Shadow should lag behind model"

    def test_shadow_initial_copy(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        ema = EMA(model, decay=0.9)
        for s, p in zip(ema.shadow.parameters(), model.parameters(), strict=True):
            assert torch.allclose(s, p), "Shadow should match model initially"

    def test_ema_decay_zero(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        ema = EMA(model, decay=0.0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        ema.update(model)
        for s, p in zip(ema.shadow.parameters(), model.parameters(), strict=True):
            assert torch.allclose(s, p), "With decay=0, shadow should match model"

    def test_ema_context_manager(self) -> None:
        model = WaveformUNet(in_channels=1, base_channels=8, depth=2)
        ema = EMA(model, decay=0.9)
        with ema.average_parameters():
            pass
        x = make_dummy(batch=1, samples=8192)
        with torch.no_grad():
            y = ema.shadow(x)
        assert y.shape == x.shape
