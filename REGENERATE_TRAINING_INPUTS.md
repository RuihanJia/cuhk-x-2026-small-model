# Regenerate training inputs locally

The adapter builds indexes, image geometry and sensor arrays from
`Training/data/HAR/data/`, using the original training labels. Generated files
are stored in a local working directory. The operation performs preprocessing
with fixed models and zero training updates.
The cohort checks target the frozen official dataset: 2,931 paired Depth/IR
clips, 2,891 thermal clips, 2,543 valid arm clips, and 18 training users.

Use the pinned Python environment, including the separately recorded pose
requirements (`opencv-python-headless==4.10.0.84`, Ultralytics 8.3.0).
The exact pose safe state is inside the selected model container at
`checkpoints/POSE.pt`; extract that file from the corresponding model container. Its expected SHA-256 is
`bc709bb30a90c8f230b0cb6288cecaf852cab29a2578752e1a43004a9ad9bdb4`.
An upstream `.pt` checkpoint has a different format and hash.

```bash
python src/training/regenerate_inputs.py \
  --data-dir /data/Small-Model-Track \
  --output /work/training_inputs \
  --pose-state /work/inference_assets/checkpoints/POSE.pt \
  --workers 4 --max-minutes 30
```

The output includes `frame_index.json`, `thermal_index.json`,
`arm_acceleration_t256.npz`, `device_acceleration_t256.npz`,
`DEVICE_CACHE_READY.json`, `conflict_weights.json`, `POSE_REGIONS.json` and
`POSE_ADAPTER_INPUTS.json`. It recomputes the pose detector on eight fixed IR
frames per clip. Labels are never passed to the detector. Only the overlap
quality rule uses original training labels, comparing clips from the same user.

`--stage indexes|sensors|conflicts|pose` runs individual stages. Run `conflicts`
before `pose`, because the final pose adapter input also includes those weights.
`--reference-inputs /local/historical/POSE_ADAPTER_INPUTS.json` enables optional
comparison against preserved original rows, conflict weights and pose boxes.
The reference enables a post-generation comparison. The completion audit lists
pose mismatches for inspection. Hardware and package versions can affect detections. `POSE_PARTIAL.json` supports
resumption of an interrupted pose stage using the same data and configuration.
Create `STOP_REQUESTED` in the output folder to request a bounded stop.

The training adapters accept these files without requiring historical home paths:

```bash
python src/training/train_video.py --kind base \
  --data-dir /data/Small-Model-Track --generated-inputs /work/training_inputs \
  --pretrained /weights/vit_b_k710_dl_from_giant.pth \
  --output /work/base --check-only

python src/training/train_auxiliary.py --kind imu \
  --data-dir /data/Small-Model-Track --generated-inputs /work/training_inputs \
  --runtime-dir src/solutions/dual_mirror/code/runtime \
  --output /work/imu --check-only

python src/training/train_auxiliary.py --kind thermal \
  --data-dir /data/Small-Model-Track --generated-inputs /work/training_inputs \
  --runtime-dir src/solutions/dual_mirror/code/runtime \
  --pretrained /weights/X3D_M.pyth --output /work/thermal --check-only
```

CONTROL4 uses the same generated input directory and the full raw16 parent:

```bash
python src/training/train_control4.py --data-dir /data/Small-Model-Track \
  --generated-inputs /work/training_inputs \
  --pretrained /weights/vit_b_k710_dl_from_giant.pth \
  --parent /work/base/model_fp16.pt --output /work/control4 --check-only
```

For a newly trained parent, add `--allow-retrained-parent`. The check loads the
parent model and one generated training sample with zero optimizer updates.

These commands load the inputs and models with zero updates. Remove
`--check-only` to train using the selected --max-hours budget.
The adapters preserve the original scientific protocol hash and separately record
regenerated input provenance. Generated input hashes are recorded separately from the original protocol. Original protocols and selected checkpoints stay intact.

Calibration can be rebuilt from source-user OOF predictions using explicit paths:

```bash
python src/training/reproduce_calibration.py --solution dual_mirror \
  --source-npz /work/oof/dual_mirror_source.npz \
  --base-source-npz /work/oof/source_oof_v2.npz \
  --output /work/calibrators/dual_mirror.json
```

The source arrays contain 4,768 rows for 2,931 distinct training clips. These arrays
come from source-user models that exclude their prediction users. The calibration
command consumes those arrays. Source-user training and prediction export use the splits and endpoints in
the retained source protocols.
