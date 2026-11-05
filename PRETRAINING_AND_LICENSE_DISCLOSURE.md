# Pretraining and model sources

Team han jack72819 · Ruihan Jia and Chang Su · University of Chinese Academy of Sciences

The following records identify the upstream initializations and final model
components. The runtime records identify the initialization sources and hashes. The
competition applies its rule 5(a)(3) exception to identified pretrained models.

| Component | Origin and use | Frozen evidence |
|---|---|---|
| VideoMAEv2 Base | Public vision-only K710-distilled Base checkpoint; adapted to 40 labels using competition training users; no teacher or language model in inference | SHA256 8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec |
| VideoMAEv2 Small | Public vision-only K710-distilled Small checkpoint, used only by dual_mirror | SHA256 24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172 |
| Thermal X3D-M | Kinetics-400 pretrained CNN, fine-tuned on native thermal pseudocolor images | SHA256 eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67 |
| Pose locator | Official YOLO11n-pose, COCO-Pose; fixed label-independent person/head/arm geometry; counted in submitted model size | Original SHA256 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0; final state SHA256 bc709bb30a90c8f230b0cb6288cecaf852cab29a2578752e1a43004a9ad9bdb4 |
| IMU | Locally initialized temporal CNN; original competition training labels | No external initialization |

## Public retrieval

- Base: https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth
- Small: https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_s_k710_dl_from_giant.pth
- X3D-M: https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth
- Pose: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt

K710, Kinetics-400 and COCO-Pose supply the upstream pretraining data. The submitted pipeline runs the adapted vision and inertial models. Challenge-specific training and calibration use the organizer-provided training labels.

## Licenses

VideoMAE2 weight metadata records Apache-2.0. OpenGVLab's VideoMAEv2 author
source uses MIT, while the retained VideoMAE implementation uses CC BY-NC 4.0.
Ultralytics uses AGPL-3.0. PyTorchVideo source uses Apache-2.0.
MODEL_LICENSES.md lists each model component and its upstream documentation. THIRD_PARTY_NOTICES.md provides implementation-source notices.
