# Model components and upstream terms

| Component | Initialization and final form | Upstream license documentation |
|---|---|---|
| VideoMAEv2 Base | K710-distilled Base; 40-class fine-tuning; six-bit or eight-bit storage | OpenGVLab/VideoMAE2 revision 706cc172d65ebd4dedbee3f9c0183a93df9fa125: Apache-2.0 weight metadata |
| VideoMAEv2 Small | K710-distilled Small; 40-class fine-tuning; eight-bit storage | Same fixed-revision Apache-2.0 weight metadata |
| Thermal X3D-M | Official Kinetics-400 X3D_M.pyth; competition fine-tuning and FP16 storage | PyTorchVideo Apache-2.0 source license and official Model Zoo |
| YOLO11n-pose | Official Ultralytics v8.3.0 checkpoint; FP16 state | Ultralytics AGPL-3.0 code and model terms |
| IMU CNN | Random initialization and competition training | Apache-2.0 original team contribution |
| Fusion parameters | Training-user OOF calibration | Apache-2.0 original team contribution |

The VideoMAE transformer implementation uses CC BY-NC 4.0. Code notices are
listed in THIRD_PARTY_NOTICES.md. The final model containers retain the original
competition checkpoint bytes.

## Initialization hashes

- Base: `8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`
- Small: `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`
- X3D-M: `eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67`
- YOLO11n-pose: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`
- Final pose state: `bc709bb30a90c8f230b0cb6288cecaf852cab29a2578752e1a43004a9ad9bdb4`

## Sources

- https://huggingface.co/OpenGVLab/VideoMAE2/tree/706cc172d65ebd4dedbee3f9c0183a93df9fa125
- https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html
- https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth
- https://github.com/ultralytics/assets/releases/tag/v8.3.0
- https://www.ultralytics.com/license

Each runtime release.json identifies its component checkpoints; artifacts/*/CONTAINER.json
identifies each complete alternative. Full license texts accompany the sources
and model release in MODEL_LICENSE_BUNDLE.zip.
