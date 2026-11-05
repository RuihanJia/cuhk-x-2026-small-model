#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <data_dir> <output_csv> [submission_id]" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_CSV="$2"
TARGET_SUBMISSION_ID="${3:-primary}"
PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# DATA_DIR is selected by the Organizing Committee at runtime. It contains the
# relevant track's official label-free Testing data in the published layout.
# Do not replace it with a hard-coded participant or server path.

if [ ! -d "$DATA_DIR" ]; then
  echo "Input data directory does not exist: $DATA_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT_CSV")"

# With two arguments, TARGET_SUBMISSION_ID is "primary" and the implementation
# must reproduce primary_verification_submission_id from submission_info.yaml.
# If a third argument is supplied, reproduce that Selected Submission ID.
# Replace the command below if your implementation uses another entry point.
exec python3 "$PACKAGE_ROOT/src/predict.py" \
  --data-dir "$DATA_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --submission-id "$TARGET_SUBMISSION_ID" \
  --config "$PACKAGE_ROOT/configs/final.yaml"
