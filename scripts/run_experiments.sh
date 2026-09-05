#!/usr/bin/env bash
# Full experiment ladder. Run from ml-service/. Every step writes results/<name>.json; scripts/report.py renders docs/results.md.
# DATA_2016 / DATA_2018 default to the synthetic files; point them at RML2016.10a_dict.pkl / GOLD_XYZ_OSC.0001_1024.hdf5 for real RadioML.
set -euo pipefail
cd "$(dirname "$0")/../ml-service"
D16=${DATA_2016:-data/synth_2016.pkl}; D18=${DATA_2018:-data/synth_2018.hdf5}; EP=${EPOCHS:-30}
run() { echo "=== $*"; python "$@"; }
run train.py pretrain --data "$D16" --epochs "$EP" --out ckpt/pre.pt --name pretrain
run train.py finetune --data "$D18" --init ckpt/pre.pt --epochs "$EP" --out ckpt/ft.pt --name ft_pre_100
run train.py finetune --data "$D18" --epochs "$EP" --out ckpt/scratch.pt --name ft_scratch_100
for f in 0.1 0.01; do
  run train.py finetune --data "$D18" --init ckpt/pre.pt --frac $f --epochs "$EP" --out ckpt/ft_pre_$f.pt --name ft_pre_$f
  run train.py finetune --data "$D18" --frac $f --epochs "$EP" --out ckpt/ft_scratch_$f.pt --name ft_scratch_$f
done
run ssl_byol.py --data "$D18" --epochs 20 --out ckpt/byol.pt --name byol
for f in 0.1 0.01; do run train.py finetune --data "$D18" --init ckpt/byol.pt --frac $f --epochs "$EP" --out ckpt/ft_byol_$f.pt --name ft_byol_$f; done
run train.py fewshot --data "$D18" --init ckpt/ft.pt --shots 5 --name fewshot_ft
run train.py fewshot --data "$D18" --init ckpt/scratch.pt --shots 5 --name fewshot_scratch
run train.py fewshot --data "$D18" --shots 5 --name fewshot_random_init
run train_sei.py --data "$D18" --init ckpt/ft.pt --name sei
run train.py finetune --data "$D18" --arch vit --epochs 15 --bs 128 --out ckpt/vit.pt --name ft_vit_100
echo "ladder complete"
