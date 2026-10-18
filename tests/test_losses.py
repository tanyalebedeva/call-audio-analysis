import torch
import torch.nn.functional as F

from callaudio import losses as L


def test_si_sdr_perfect_and_scale_invariant():
    ref = torch.randn(2, 16000)
    assert (L.si_sdr(ref, ref) > 60).all()
    est = ref + 0.3 * torch.randn(2, 16000)
    assert torch.allclose(L.si_sdr(3.0 * est, ref), L.si_sdr(est, ref), atol=1e-3)


def test_si_sdr_noisier_is_lower():
    ref = torch.randn(1, 16000)
    noise = torch.randn(1, 16000)
    assert L.si_sdr(ref + 0.1 * noise, ref) > L.si_sdr(ref + 1.0 * noise, ref)


def test_separation_losses_zero_for_perfect_estimate():
    ref = torch.randn(2, 16000)
    assert L.SpectralL1Loss()(ref, ref).item() < 1e-6
    assert L.MultiResolutionSTFTLoss()(ref, ref).item() < 1e-5


def test_focal_equals_ce_when_gamma_zero():
    logits, target = torch.randn(8, 4), torch.randint(0, 4, (8,))
    assert torch.allclose(L.FocalLoss(gamma=0.0)(logits, target), F.cross_entropy(logits, target), atol=1e-6)


def test_class_weights_favor_rare_class():
    w = L.class_weights([0, 0, 0, 0, 1], num_classes=2)
    assert w[1] > w[0]


def test_vae_kl_zero_for_standard_normal():
    mu, logvar = torch.zeros(4, 8), torch.zeros(4, 8)
    assert torch.allclose(L.kl_divergence(mu, logvar), torch.zeros(4))


def test_beta_schedule_warms_up():
    assert L.beta_schedule(0, 0.0, 1.0, 5) == 0.0
    assert abs(L.beta_schedule(2, 0.5, 1.0, 5) - 0.5) < 1e-9
    assert L.beta_schedule(10, 0.0, 1.0, 5) == 1.0
