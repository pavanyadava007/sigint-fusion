# ml-service

Modulation classifier (PyTorch training, ONNX Runtime serving), classical SIGINT DSP, multi-sensor fusion, Kafka consumer,
sensor simulator and load benchmark.

## Layout
| Path | Purpose |
|---|---|
| `data/synth_mod.py` | 24-class synthetic modulator writing RadioML 2016/2018 file layouts (`python -m data.synth_mod --out data/synth_2018.hdf5`) |
| `data/radioml.py` | chunked loaders for RadioML 2016.10a (pickle) and 2018.01A (HDF5), also read the synthetic files |
| `data/sei.py` | synthetic hardware fingerprints for specific emitter identification; ORACLE loader |
| `models/resnet1d.py` | length-agnostic ResNet-1D with swappable head and `embed()` for prototypes (served model) |
| `models/vit_spec.py` | STFT frontend + timm ViT-tiny (comparison model) |
| `sigproc/dsp.py` | Welch PSD, STFT spectrogram, vectorised CA-CFAR, DBSCAN PDW deinterleaver with PRI typing |
| `sigproc/pdw_synth.py` | synthetic radar pulse trains with ground truth |
| `fusion/associate.py` | Kalman + Hungarian emitter tracker producing the EOB (`step_multi` for several sensors per step) |
| `train.py` | `pretrain` / `finetune` / `fewshot`; writes `results/<name>.json` with provenance and exports ONNX with parity check |
| `ssl_byol.py`, `train_sei.py` | BYOL self-supervised pretraining; SEI experiment (12 train devices, 4 unseen, 5-shot) |
| `app.py` | FastAPI: `/classify /detect /spectrogram /deinterleave /synth` + console feeds `/detections /spectrum /eob /stats /sensors /reports` + `/metrics` |
| `consumer.py` | Kafka `iq-chunks` + `pdw-batches` -> classifier / deinterleaver -> fusion -> Postgres |
| `sensor_sim.py` | scenario of 7 emitters observed by 3 COMINT sensors and 1 R-ESM sensor, posts to the gateway |
| `benchmark/load.py` | async `/classify` load test -> `results/benchmark_classify.json` |
| `tests/` | 40+ unit and service tests (tiny ONNX model exported in-test; DB endpoints must degrade to 503) |

## Experiments
`../scripts/run_experiments.sh` runs the whole ladder; single steps:
```bash
python train.py pretrain --data data/synth_2016.pkl --out ckpt/pre.pt
python train.py finetune --data data/synth_2018.hdf5 --init ckpt/pre.pt --out ckpt/ft.pt          # -> ckpt/ft.onnx (served)
python train.py finetune --data data/synth_2018.hdf5 --frac 0.1 --out ckpt/scratch10.pt --name ft_scratch_0.1
python ssl_byol.py --data data/synth_2018.hdf5 --out ckpt/byol.pt
python train.py fewshot  --data data/synth_2018.hdf5 --init ckpt/ft.pt --shots 5
python train_sei.py --data data/synth_2018.hdf5 --init ckpt/ft.pt
python train.py finetune --data data/synth_2018.hdf5 --arch vit --epochs 15 --bs 128 --out ckpt/vit.pt
```
Point `--data` at `RML2016.10a_dict.pkl` / `GOLD_XYZ_OSC.0001_1024.hdf5` to use the real RadioML sets (loaders are chunked, the
21 GB file is never fully materialised). Add `--mlflow` to log to MLflow.

## Serve and test
```bash
uvicorn app:app --port 8000            # MODEL=ckpt/ft.onnx PG=postgresql://...
python consumer.py                     # KAFKA=localhost:29092 when running outside compose
python sensor_sim.py --gateway http://localhost:8080 --rate 5
python benchmark/load.py --n 2000 --conc 32
CUDA_VISIBLE_DEVICES= python -m pytest -q tests fusion
```
