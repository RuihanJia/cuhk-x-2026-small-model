# Model provenance

Both model assets are the original selected competition containers.

| Alternative | Selected submission | Complete container bytes |
|---|---|---:|
| dual_mirror | 56258094 | 99,964,493 |
| weight_average | 56258110 | 99,431,328 |

artifacts/*/CONTAINER.json records complete-container hashes. Each runtime's
release.json records component checkpoint hashes, calibration and inference
settings. REPRODUCTION_RESULTS.json records the selected output hashes and
offline batch measurements. The PAIR header records the video packing stage;
the runtime release configuration records final calibration and view settings.
