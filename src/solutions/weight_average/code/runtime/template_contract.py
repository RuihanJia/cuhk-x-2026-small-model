"""Independent fixed 405-path order, extracted from the original public test manifest."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
FROZEN_PATHS_SHA256='8593609DDEE7D40453F7FA45754D0D93ADCFAE3FE82835064CE01BBECED2C6A3'

def expected_paths():
    path=ROOT/'FROZEN_SAMPLE_PATHS.json'
    if hashlib.sha256(path.read_bytes()).hexdigest().upper()!=FROZEN_PATHS_SHA256:
        raise ValueError('FROZEN_TEMPLATE_HASH_MISMATCH')
    values=json.loads(path.read_text(encoding='utf-8'))['normalized_output_paths']
    if len(values)!=405 or len(set(values))!=405:
        raise ValueError('FROZEN_TEMPLATE_ROW_COUNT_OR_DUPLICATION')
    return values

def verify_paths(paths):
    if list(paths)!=expected_paths():
        raise ValueError('ORIGINAL_405_TEMPLATE_PATH_ORDER_MISMATCH')
