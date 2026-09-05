"""Service tests: FastAPI endpoints against a tiny ONNX model exported in the test, and the consumer's pure processing functions."""

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from data.synth_mod import make_sample
from models.resnet1d import ResNet1D
from sigproc.pdw_synth import synth_pdw


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    path = str(d / "tiny.onnx")
    m = ResNet1D(4, width=8, depth=2, emb=16).eval()
    torch.onnx.export(m, torch.zeros(1, 2, 256), path, input_names=["iq"], output_names=["logits"],
                      dynamic_axes={"iq": {0: "batch", 2: "length"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    json.dump(["BPSK", "QPSK", "FM", "GMSK"], open(path + ".classes.json", "w"))
    return path


@pytest.fixture(scope="module")
def client(onnx_model):
    import app as app_module

    app_module.model = app_module.Model(onnx_model)
    app_module.PG = "postgresql://nobody:nobody@127.0.0.1:1/none"  # unreachable on purpose
    with TestClient(app_module.app) as c:
        yield c


def test_health_reports_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["classes"] == 4


def test_classify_returns_sorted_probs(client):
    x = make_sample("QPSK", 1024, 20, 0)
    r = client.post("/classify", json=dict(i=x[:, 0].tolist(), q=x[:, 1].tolist(), top_k=3))
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3 and preds[0]["prob"] >= preds[1]["prob"] and r.json()["latency_ms"] >= 0


def test_classify_validation(client):
    assert client.post("/classify", json=dict(i=[0.0] * 100, q=[0.0] * 99)).status_code == 422
    assert client.post("/classify", json=dict(i=[0.0] * 10, q=[0.0] * 10)).status_code == 422


def test_detect_and_spectrogram(client):
    n = np.arange(1024)
    rng = np.random.default_rng(1)
    body = dict(i=(np.cos(2 * np.pi * 0.1 * n) + 0.05 * rng.standard_normal(1024)).tolist(),
                q=(np.sin(2 * np.pi * 0.1 * n) + 0.05 * rng.standard_normal(1024)).tolist())  # narrowband tone + noise
    d = client.post("/detect", json=body).json()
    assert len(d["freqs"]) == len(d["levels_db"]) == len(d["mask"]) and d["n_detections"] >= 1
    s = client.post("/spectrogram", json=body).json()
    assert len(s["db"]) == len(s["freqs"]) and len(s["db"][0]) == len(s["times"])


def test_deinterleave_endpoint(client):
    pdw, truth, _ = synth_pdw(n_emitters=3, seed=2)
    r = client.post("/deinterleave", json=dict(pdw=pdw.tolist())).json()
    assert len(r["emitters"]) == 3 and len(r["labels"]) == len(pdw)


def test_synth_endpoint(client):
    r = client.post("/synth", json=dict(modulation="8PSK", snr_db=5, length=512))
    assert r.status_code == 200 and len(r.json()["i"]) == 512 and r.json()["synthetic"] is True
    assert client.post("/synth", json=dict(modulation="nope")).status_code == 400


def test_db_endpoints_degrade_to_503(client):
    assert client.get("/detections").status_code == 503
    assert client.get("/eob").status_code == 503


def test_metrics(client):
    assert b"ml_requests_total" in client.get("/metrics").content


def test_consumer_processing(onnx_model):
    import consumer

    clf = consumer.Classifier(onnx_model)
    x = make_sample("FM", 512, 15, 3)
    msgs = [dict(sensor_id="s1", rf_mhz=156.8, aoa=40.0, i=x[:, 0].tolist(), q=x[:, 1].tolist())] * 3
    dets, spectra, fus = consumer.process_iq(msgs, clf)
    assert len(dets) == len(spectra) == len(fus) == 3 and dets[0][1] == "comint" and len(spectra[0][2]) == 256
    pdw, _, _ = synth_pdw(n_emitters=2, seed=4)
    d2, f2 = consumer.process_pdw([dict(sensor_id="esm", pdw=pdw.tolist()), dict(sensor_id="esm", pdw=[[0, 1, 2]])])
    assert len(d2) == 2 and d2[0][1] == "resm" and f2[0]["source"] == "resm"
    from fusion.associate import Fuser

    eob = Fuser().step_multi([fus, f2], 1.0)  # 3 repeated sightings of one emitter + 2 radars -> 3 tracks
    assert len(eob) == 3 and all(np.isfinite(t["rf_mhz"]) for t in eob)
