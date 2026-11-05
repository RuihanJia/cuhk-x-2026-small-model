# Offline verification

Run `tools/verify_offline.py` on Linux with a repository, release-asset directory,
official data root and a new output directory. The asset directory contains
`dual_mirror-model.pth` and `weight_average-model.pth`.

The outer process records its command and network namespace, then launches
the verifier with `unshare --net`. The inner process checks that its namespace
differs, that its sole interface is loopback, and that it has no external IPv4
route. Each inference child is checked against the inner namespace.

Before inference, the verifier creates a fresh source copy, compares its file
hashes with the input repository, checks that prediction and feature caches are
absent, and copies the two frozen model containers. It records Python, PyTorch,
CUDA, GPU and installed package versions. The existing temperature controller
monitors the inference processes. The default wall-clock limit is 30 minutes.

| File | Contents |
|---|---|
| LAUNCH.json | Actual outer command, script hash and parent namespace |
| NETWORK_NAMESPACE.json | Parent/inner namespaces, interfaces and routes |
| SOURCE_INPUT_MANIFEST.json | Hashes of input source files |
| EXECUTED_SOURCE_MANIFEST.json | Hashes of the exact execution copy |
| EXECUTION_SETUP.json | Cache check, source-copy check and model hashes |
| ENVIRONMENT.json, pip-freeze.txt | Runtime and installed package versions |
| `*-launch.json` | Child command, PID and network namespace |
| `*-telemetry.jsonl` | Temperature and controller states |
| RESULTS.json | Output hashes and elapsed times for both alternatives |
| SOURCE_FINAL_CHECK.json | End-of-run source consistency |

`--check-only` performs launch, namespace, environment and input-source checks.
The full run compares both outputs with the fixed selected-submission hashes.
Each run uses a new output directory. The temporary execution copy is removed
after the recorded checks complete. The selected output hashes are listed in REPRODUCTION_RESULTS.json.
