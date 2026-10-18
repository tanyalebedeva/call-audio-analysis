import numpy as np
import torch

from callaudio.audio import mix_at_snr, rms, telephone_channel
from callaudio.data import CommandsMelDataset, EmotionDataset, SeparationDataset
from callaudio.metrics import classification_metrics, normalize_text
from callaudio.models import ConvTasNet, ConvVAE, MelCNN, UNetMask, WaveCNN


def test_mix_at_snr_exact():
    rng = np.random.default_rng(0)
    speech, noise = rng.standard_normal(16000).astype(np.float32), rng.standard_normal(16000).astype(np.float32)
    for snr in (-5, 0, 10):
        _, scaled = mix_at_snr(speech, noise, snr)
        assert abs(20 * np.log10(rms(speech) / rms(scaled)) - snr) < 1e-3


def test_telephone_keeps_length():
    wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    assert len(telephone_channel(wav)) == 16000


def test_separation_dataset_deterministic():
    ds = SeparationDataset(synthetic=True, deterministic=True, segment_sec=1.0, snr_grid=(0, 5), epoch_size=4)
    a, b = ds[1], ds[1]
    assert torch.equal(a["mix"], b["mix"]) and a["snr"] == 5.0
    assert a["mix"].shape == a["clean"].shape == (16000,)


def test_emotion_and_commands_datasets():
    e = EmotionDataset(synthetic=True, synthetic_size=8, max_sec=2.0)
    assert e[0]["wav"].shape == (32000,) and 0 <= e[0]["label"] < 4
    c = CommandsMelDataset(synthetic=True, synthetic_size=8)
    assert c[0]["mel"].shape == (80, 64)


def test_models_keep_length_and_shapes():
    x = torch.randn(2, 20000)  # длина не кратна шагу STFT и свёрток
    assert UNetMask()(x).shape == x.shape
    assert ConvTasNet(blocks=2, repeats=1)(x).shape == x.shape
    assert MelCNN(num_classes=4)(x).shape == (2, 4)
    assert WaveCNN(num_classes=4).embed(x).shape == (2, 128)
    recon, mu, _ = ConvVAE(latent_dim=8)(torch.randn(2, 80, 64))
    assert recon.shape == (2, 80, 64) and mu.shape == (2, 8)


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert abs(m["wa"] - 0.75) < 1e-9 and abs(m["ua"] - 0.75) < 1e-9
    assert m["confusion"] == [[1, 1], [0, 2]]


def test_normalize_text():
    assert normalize_text("Ёлка, ЗЕЛЁНАЯ!") == "елка зеленая"
