#!/usr/bin/env bash
# Full experiment ladder. Run from anywhere. Every step writes ml-service/results/<name><TAG>.json; scripts/report.py renders docs/results.md.
# DATA_2016 / DATA_2018 default to the synthetic files; point them at RML2016.10a_dict.pkl / GOLD_XYZ_OSC.0001_1024.hdf5 for real RadioML.
# TAG defaults to "" for synthetic data and "_real" otherwise; it suffixes result names and selects the checkpoint dir (ckpt/ or ckpt/real/),
# so synthetic and real runs coexist and the report renders both. PY overrides the interpreter (default: ../.venv/bin/python if present).
set -euo pipefail
cd "$(dirname "$0")/../ml-service"
D16=${DATA_2016:-data/synth_2016.pkl}; D18=${DATA_2018:-data/synth_2018.hdf5}; EP=${EPOCHS:-30}
MPC=${MAX_PER_CLASS:-4000}   # frames per class kept from the 2018 file (real 2018.01A has ~106k per class; 12000 is a good GPU-hour budget)
if [ -z "${TAG+x}" ]; then case "$D18" in *synth*) TAG="";; *) TAG="_real";; esac; fi
C=ckpt${TAG:+/real}; mkdir -p "$C"
PY=${PY:-$([ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python)}
run() { echo "=== $*"; "$PY" "$@"; }
run train.py pretrain --data "$D16" --epochs "$EP" --out "$C/pre.pt" --name "pretrain$TAG"
run train.py finetune --data "$D18" --init "$C/pre.pt" --epochs "$EP" --max_per_class "$MPC" --out "$C/ft.pt" --name "ft_pre_100$TAG"
run train.py finetune --data "$D18" --epochs "$EP" --max_per_class "$MPC" --out "$C/scratch.pt" --name "ft_scratch_100$TAG"
for f in 0.1 0.01; do
  run train.py finetune --data "$D18" --init "$C/pre.pt" --frac $f --epochs "$EP" --max_per_class "$MPC" --out "$C/ft_pre_$f.pt" --name "ft_pre_$f$TAG"
  run train.py finetune --data "$D18" --frac $f --epochs "$EP" --max_per_class "$MPC" --out "$C/ft_scratch_$f.pt" --name "ft_scratch_$f$TAG"
done
run ssl_byol.py --data "$D18" --epochs 20 --out "$C/byol.pt" --name "byol$TAG"
for f in 0.1 0.01; do run train.py finetune --data "$D18" --init "$C/byol.pt" --frac $f --epochs "$EP" --max_per_class "$MPC" --out "$C/ft_byol_$f.pt" --name "ft_byol_$f$TAG"; done
run train.py fewshot --data "$D18" --init "$C/ft.pt" --shots 5 --name "fewshot_ft$TAG"
run train.py fewshot --data "$D18" --init "$C/scratch.pt" --shots 5 --name "fewshot_scratch$TAG"
run train.py fewshot --data "$D18" --shots 5 --name "fewshot_random_init$TAG"
run train_sei.py --data "$D18" --init "$C/ft.pt" --out "$C/sei.pt" --name "sei$TAG"
run train.py finetune --data "$D18" --arch vit --epochs 15 --bs 128 --max_per_class "$MPC" --out "$C/vit.pt" --name "ft_vit_100$TAG"
echo "ladder complete (tag='$TAG', checkpoints in $C)"
