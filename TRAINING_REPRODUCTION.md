# Training recipes and reproduction

## 1. Generate the training inputs

Follow REGENERATE_TRAINING_INPUTS.md. The generated directory contains the
2,931-clip video cohort, thermal index, native IMU features, pose geometry and
same-user overlap weights. It is passed to each adapter with `--generated-inputs`.

## 2. Full-data video models

```bash
python src/training/train_video.py --kind base --data-dir DATA_DIR \
  --generated-inputs /work/training_inputs \
  --pretrained /weights/vit_b_k710_dl_from_giant.pth --output /work/base --check-only
python src/training/train_video.py --kind small --data-dir DATA_DIR \
  --generated-inputs /work/training_inputs \
  --pretrained /weights/vit_s_k710_dl_from_giant.pth --output /work/small --check-only
```

The fixed recipe uses 16 epochs, seed 20260915, AdamW, effective batch 16,
backbone learning rate 1e-4, head rate 5e-4, layer decay 0.75 and two warm-up
epochs. The original protocol and worker are retained in reproduction_tree.
Removing `--check-only` starts the registered recipe. The adapter records the
scientific protocol hash and the actual generated-input hash.

## 3. Thermal and IMU

```bash
python src/training/train_auxiliary.py --kind thermal --data-dir DATA_DIR \
  --generated-inputs /work/training_inputs \
  --runtime-dir src/solutions/dual_mirror/code/runtime \
  --pretrained /weights/X3D_M.pyth --output /work/thermal --check-only
python src/training/train_auxiliary.py --kind imu --data-dir DATA_DIR \
  --generated-inputs /work/training_inputs \
  --runtime-dir src/solutions/dual_mirror/code/runtime --output /work/imu --check-only
```

Thermal uses X3D-M, 20 epochs, seed 2026091003, AdamW at 3e-4 and effective
batch 8. IMU uses 60 epochs, batch 64, AdamW at 1e-3, five device slots and
0.25 training-time dropout on the chest and leg slots. The original native
resampling and normalizer rules are retained with the workers.

## 4. CONTROL4 and the secondary model

```bash
python src/training/train_control4.py --data-dir DATA_DIR \
  --generated-inputs /work/training_inputs \
  --pretrained /weights/vit_b_k710_dl_from_giant.pth \
  --parent /work/base/model_fp16.pt --output /work/control4 --check-only
```

CONTROL4 starts from the raw16 Base endpoint with a fresh optimizer for four
epochs. Sampling continues at the corresponding epoch offset. The two FP16
endpoints are averaged with weight 0.5 in FP32, cast to FP16, then packed with
eight-bit video matrices. For a newly trained parent, add `--allow-retrained-parent`.
The zero-update check loads the parent and a generated training sample.

## 5. Source-user calibration

Source models exclude their prediction users during branch training. Original
source jobs are retained in videomaev2_base_source, videomaev2_source and the
corresponding auxiliary protocols. Their OOF export supplies 4,768 rows for
2,931 distinct training clips. Calibration uses user-balanced loss, a global
mixture and classwise correction with L2 coefficient 0.1.

```bash
python src/training/reproduce_calibration.py --solution dual_mirror \
  --source-npz /work/oof/dual_mirror_source.npz \
  --base-source-npz /work/oof/source_oof_v2.npz --output /work/dual_mirror.json
python src/training/reproduce_calibration.py --solution weight_average \
  --source-npz /work/oof/weight_average_source.npz \
  --base-source-npz /work/oof/source_oof_v2.npz --output /work/weight_average.json
```

## Outputs

The adapters write execution configuration, input hashes, checkpoints and
progress to the selected output directory. --check-only loads the inputs and
models with zero optimizer updates. The complete training command uses the
same arguments with --check-only removed.

Calibration inputs contain source-fold identifiers, clip identifiers, users,
targets, modality availability and branch predictions. Source-user models use
the train/held-user splits in the retained source protocols. Training outputs,
sensor caches and OOF arrays are stored in the selected working directories.
