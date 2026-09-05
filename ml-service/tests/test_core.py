import numpy as np
import pytest
import torch

from data.radioml import MODS_2018
from data.synth_mod import CONSTELLATIONS, make_sample
from models.resnet1d import ResNet1D
from sigproc.dsp import ca_cfar, deinterleave_pdw, psd, spectrogram
from sigproc.pdw_synth import purity_completeness, synth_pdw


def test_model_length_agnostic():
    m = ResNet1D(24).eval()
    assert m(torch.zeros(2, 2, 128)).shape == (2, 24) and m(torch.zeros(2, 2, 1024)).shape == (2, 24)


def test_transfer_head_swap():
    m = ResNet1D(11)
    m.reset_head(21)
    m.freeze_backbone()
    assert m.head.out_features == 21 and not next(m.backbone.parameters()).requires_grad


@pytest.mark.parametrize("mod", MODS_2018)
def test_synth_all_modulations_unit_power(mod):
    x = make_sample(mod, 512, 10, 1)
    assert x.shape == (512, 2) and np.isfinite(x).all()
    assert abs((x**2).sum(1).mean() - 1.0) < 1e-3


def test_synth_constellation_sizes():
    for name, c in CONSTELLATIONS.items():
        m = {"OOK": 2, "BPSK": 2, "QPSK": 4}.get(name) or int("".join(ch for ch in name if ch.isdigit()))
        assert len(c) == m, name


def test_synth_deterministic():
    assert np.array_equal(make_sample("QPSK", 256, 5, 42), make_sample("QPSK", 256, 5, 42))


def _cfar_reference(x_db, guard=2, train=8, pfa=1e-3):
    N = len(x_db)
    lin = 10 ** (x_db / 10)
    alpha = 2 * train * (pfa ** (-1 / (2 * train)) - 1)
    det = np.zeros(N, bool)
    for i in range(N):
        lo, hi = max(0, i - guard - train), min(N, i + guard + train + 1)
        cells = np.r_[lin[lo : max(0, i - guard)], lin[i + guard + 1 : hi]]
        if len(cells) and lin[i] > alpha * cells.mean():
            det[i] = True
    return det


def test_cfar_matches_loop_reference_and_detects_tone():
    rng = np.random.default_rng(0)
    p = rng.normal(-30, 3, 256)
    p[100] = 0.0
    assert ca_cfar(p)[100]
    assert np.array_equal(ca_cfar(p), _cfar_reference(p))


def test_cfar_false_alarm_rate_is_low_on_noise():
    rng = np.random.default_rng(1)
    noise = 10 * np.log10(rng.exponential(1.0, 20000))
    assert ca_cfar(noise, pfa=1e-3).mean() < 5e-3


def test_deinterleave_quality():
    pdw, truth, _ = synth_pdw(n_emitters=4, seed=1)
    labels, em = deinterleave_pdw(pdw)
    q = purity_completeness(labels, truth)
    assert q["purity"] > 0.9 and q["completeness"] > 0.9 and len(em) == 4


def test_deinterleave_empty():
    labels, em = deinterleave_pdw(np.zeros((0, 4)))
    assert len(labels) == 0 and em == {}


def test_psd_and_spectrogram_shapes():
    x = make_sample("FM", 1024, 20, 0).T
    f, P = psd(x)
    assert f.shape == P.shape == (256,)
    f2, t, S = spectrogram(x)
    assert S.shape == (len(f2), len(t)) and np.isfinite(S).all()
