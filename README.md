# CUHK-X Challenge 2026 — Small Model Track

Team **han jack72819** · Ruihan Jia and Chang Su · University of Chinese Academy of Sciences

## Method and results

The primary solution combines VideoMAEv2 Base and Small infrared video encoders
with thermal X3D-M and an IMU temporal CNN. A shared YOLO11n-pose model supplies
head-and-arm crop geometry. Base uses six-bit weight storage and Small uses
eight-bit storage; inference reconstructs floating-point weights. Fixed temporal
and mirrored views are combined through source-user calibration and classwise fusion.

| Alternative | Selected submission | Public accuracy | Private accuracy | Model bytes |
|---|---:|---:|---:|---:|
| dual_mirror | 56258094 | 0.88059 | 0.87745 | 99,964,493 |
| weight_average | 56258110 | 0.86069 | 0.86274 | 99,431,328 |

Each alternative runs independently. These scores are the retained competition results.

## Install

Verified platform: Ubuntu 24.04 x86_64 under WSL2, Python 3.12, PyTorch 2.12.1+cu130,
and one RTX 4070 Laptop GPU with 8 GB memory. Use a CUDA-13-compatible NVIDIA driver.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
```

The full package lock is in environment/linux-lock.txt. The official dataset
is obtained through the competition.

## Models and inference

See [DOWNLOAD_MODELS.md](DOWNLOAD_MODELS.md) for asset hashes and placement.
Model release: [https://github.com/RuihanJia/cuhk-x-2026-small-model/releases/tag/v1.0.0](https://github.com/RuihanJia/cuhk-x-2026-small-model/releases/tag/v1.0.0).

Place the two model containers at `artifacts/dual_mirror/model.pth` and
`artifacts/weight_average/model.pth`, then run:

```bash
bash inference.sh /path/to/Small-Model-Track /work/primary.csv
bash inference.sh /path/to/Small-Model-Track /work/secondary.csv 56258110
```

The entry verifies the model container and its checkpoint hashes. It discovers
the supplied Testing recordings and computes IR, thermal and IMU predictions
from raw inputs. Each model container is a ZIP_STORED bundle loaded by this entry.

## Reproduction checks

Both 405-row outputs match the selected competition submissions byte for byte.
The instrumented offline runs took 476.45 s for dual_mirror and 313.57 s for
weight_average. The report cites earlier batch runs of 443.16 s and 308.90 s.
[REPRODUCTION_RESULTS.json](REPRODUCTION_RESULTS.json) records the checks.

The portable verifier creates a network namespace, checks its interfaces and
routes, copies source into a fresh execution directory, and records source hashes,
model hashes, environment versions and both inference commands:

```bash
python tools/verify_offline.py --repo "$PWD" --models /work/release_assets \
  --data /data/Small-Model-Track --output /work/verification --max-minutes 30
```

Linux permission to create a network namespace is required; WSL users can run
the command as root. [OFFLINE_VERIFICATION.md](OFFLINE_VERIFICATION.md) lists the
output records and the checks performed.

## Training

[REGENERATE_TRAINING_INPUTS.md](REGENERATE_TRAINING_INPUTS.md) describes raw Training
index generation, IMU resampling, conflict weights and pose geometry. All training
adapters accept the generated input directory, including the CONTROL4 stage used
by weight_average. Source protocols retain their original fixed endpoints.

The training guide lists fixed recipes, user splits, source-user calibration
inputs and output files. See [TRAINING_REPRODUCTION.md](TRAINING_REPRODUCTION.md).

## Provenance and licenses

The inference sources and both model containers retain the competition versions.
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) identifies each selected container,
its component checkpoints and output hashes.

Original contributions use Apache License 2.0 within [LICENSE_SCOPE.md](LICENSE_SCOPE.md).
Third-party code and models retain the terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The report uses CC BY 4.0, © 2026 The Authors.
