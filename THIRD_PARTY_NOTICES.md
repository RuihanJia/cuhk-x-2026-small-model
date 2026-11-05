# Third-party software and model notices

Both selected alternatives retain the original inference implementation. The
two runtimes contain duplicate copies of some upstream code. The following
scope applies to both copies and corresponding retained training-tree copies.

| Component | Version/source | License evidence | Role |
|---|---|---|---|
| Retained VideoMAE implementation | `upstream_modeling_finetune.py`, SHA-256 `ce9a11ff05c7a1b5085addf6c73b71ffb68933564164f7b250dedad91f6e29b8` | CC BY-NC 4.0, retained full license | Video transformer definitions used by the Base/Small wrappers |
| OpenGVLab VideoMAEv2 author materials | OpenGVLab/VideoMAEv2 | MIT notice retained separately | Upstream architecture and provenance; the retained VideoMAE implementation has a separate CC BY-NC notice |
| Ultralytics | 8.3.0, vendored `pose_site/ultralytics` | AGPL-3.0, full license retained | YOLO11n-pose reconstruction and inference for head/arm crop geometry |
| PyTorchVideo | 0.1.5 | Apache-2.0, vendored license | X3D-M thermal model |
| fvcore | 0.1.5.post20221221 | Apache-2.0, metadata and upstream license | PyTorchVideo support |
| iopath | 0.1.10 | MIT, vendored license | File I/O support |
| yacs | 0.1.8 | Retained LICENSE and NOTICE (Apache-2.0) | Configuration support |
| portalocker | 2.10.1 | BSD-3-Clause, vendored license | Locking support |
| tabulate | 0.9.0 | MIT, vendored license | Formatting support |
| termcolor | 2.5.0 | MIT, upstream notice | Terminal formatting |
| PyYAML source/shims | See installed PyYAML 6.0.3 lock and retained source | MIT; upstream full license supplied | YAML configuration |
| tqdm | 4.67.1 | MIT and MPL-2.0, vendored LICENCE | Progress support |
| pywin32 source/shims | Distribution metadata 311 | Upstream Mark Hammond source license (specific-file exceptions apply); metadata labels PSF | Historical Windows compatibility source; binaries excluded; Linux is the supported release verification platform |

Installed packages are listed separately in `environment/linux-lock.txt`.
Their upstream distributors provide the applicable terms. Historical local shims retain their upstream context.

## Models and initialization

| Model | Origin | Use and license handling |
|---|---|---|
| VideoMAEv2 K710 Base/Small initialization | OpenGVLab/VideoMAE2, revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125` | Vision-only distilled pretraining; retained metadata records Apache-2.0. Adapted using permitted competition training labels. |
| X3D-M initialization | PyTorchVideo Kinetics-400 model zoo | Thermal branch; retained code license/provenance. Upstream dataset terms are listed separately from software terms. |
| YOLO11n-pose | Ultralytics assets v8.3.0; COCO-Pose | Fixed image geometry; included in model-size accounting. Retains applicable upstream terms, separately identified under the organizer's pretrained-model exception. |
| IMU CNN | Team initialization | No external pretrained initialization. |

Final model containers contain learned weights, preprocessing model state,
calibration and provenance. Their exact component hashes are retained in each
runtime's `release.json` and each model's `CONTAINER.json`. The model containers
are delivered as release assets, not committed to Git.

## Source locations

- VideoMAE: https://github.com/MCG-NJU/VideoMAE
- VideoMAEv2: https://github.com/OpenGVLab/VideoMAEv2
- Pretrained VideoMAE2: https://huggingface.co/OpenGVLab/VideoMAE2
- Ultralytics 8.3.0: https://github.com/ultralytics/ultralytics/tree/v8.3.0
- PyTorchVideo 0.1.5: https://github.com/facebookresearch/pytorchvideo/tree/v0.1.5
- fvcore: https://github.com/facebookresearch/fvcore
- iopath: https://github.com/facebookresearch/iopath
- yacs: https://github.com/rbgirshick/yacs
- portalocker: https://github.com/wolph/portalocker
- tqdm: https://github.com/tqdm/tqdm

The public-source inventory records file hashes. The original signed competition verification archive is retained privately as evidence. Public model assets are listed in DOWNLOAD_MODELS.md.

## VideoMAE attribution

The retained upstream_modeling_finetune.py uses CC BY-NC 4.0.
LICENSES/VideoMAE-NOTICE.md contains the upstream attribution notices,
including Apache-2.0 material from timm/SlowFast and MIT material from BEiT.
