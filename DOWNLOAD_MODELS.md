# Model downloads

Release: https://github.com/RuihanJia/cuhk-x-2026-small-model/releases/tag/v1.0.0

Download dual_mirror-model.pth, weight_average-model.pth and SHA256SUMS.txt.
Verify each SHA-256 and place the files as follows:

```
artifacts/dual_mirror/model.pth
artifacts/weight_average/model.pth
```

| Alternative | SHA-256 | Bytes |
|---|---|---:|
| dual_mirror | `a1035dc80fcb23fa53733b9599cc9f508d78052344f69ad688fa01ba56517624` | 99,964,493 |
| weight_average | `6924957abe18fc6fcb64086ffcd9030f8aab9f0b6daa45d88531d26d9efdce41` | 99,431,328 |

The inference entry verifies and extracts each ZIP_STORED container. Each
alternative runs independently. The model components and their upstream terms
are listed in MODEL_LICENSES.md and THIRD_PARTY_NOTICES.md.
