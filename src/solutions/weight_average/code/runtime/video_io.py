"""Exact inherited sample discovery, center sampling and nearest-decodable recovery."""
from __future__ import annotations
import csv,hashlib,json,os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from PIL import Image,UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_f
from torchvision.transforms.functional import pil_to_tensor
MODALITIES=("Depth_Color","IR")

@dataclass(frozen=True)
class Sample:
    output_path: str
    sample_dir: Path

class CenterCropVideoTransform(nn.Module):
    """Exact evaluation transform used by the frozen R160 training adapter."""

    def __init__(self, *, frames: int, resize_short_side: int, crop_size: tuple[int, int], mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super().__init__()
        self.frames = frames
        self.resize_short_side = resize_short_side
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[0] != self.frames or frames.shape[1] != 3:
            raise ValueError(f'expected raw frames shaped ({self.frames}, 3, H, W), received {tuple(frames.shape)}')
        value = frames.to(dtype=torch.float32).div_(255.0)
        value = transform_f.resize(value, [self.resize_short_side], interpolation=InterpolationMode.BILINEAR, antialias=True)
        value = transform_f.center_crop(value, list(self.crop_size))
        value = transform_f.normalize(value, self.mean, self.std)
        expected = (self.frames, 3, *self.crop_size)
        if tuple(value.shape) != expected:
            raise ValueError(f'unexpected transformed frame shape {tuple(value.shape)}')
        return value.permute(1, 0, 2, 3).contiguous()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def locate_sample_root(data_dir: Path) -> Path:
    candidates = [data_dir / 'small_model_track_test', data_dir / 'data' / 'small_model_track_test', data_dir / 'Testing' / 'data' / 'small_model_track_test']
    if data_dir.name == 'small_model_track_test':
        candidates.insert(0, data_dir)
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if not existing:
        matches = sorted((path for path in data_dir.rglob('small_model_track_test') if path.is_dir()), key=lambda value: (len(value.parts), str(value)))
        existing = matches
    unique = list(dict.fromkeys((candidate.resolve() for candidate in existing)))
    if len(unique) != 1:
        raise FileNotFoundError(f'expected exactly one small_model_track_test directory under {data_dir}; found {unique}')
    return unique[0]

def locate_manifest(data_dir: Path) -> Path | None:
    candidates = [data_dir / 'test.csv', data_dir / 'test_file' / 'test.csv', data_dir / 'Testing' / 'test_file' / 'test.csv']
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        existing = sorted((path for path in data_dir.rglob('test.csv') if path.is_file()), key=lambda value: (len(value.parts), str(value)))
    unique = list(dict.fromkeys((candidate.resolve() for candidate in existing)))
    if len(unique) > 1:
        raise ValueError(f'ambiguous test.csv files under {data_dir}: {unique}')
    return unique[0] if unique else None

def normalized_manifest_name(value: str) -> str:
    normalized = value.strip().replace('\\', '/').rstrip('/')
    name = normalized.split('/')[-1]
    if not name:
        raise ValueError(f'invalid manifest path: {value!r}')
    return name

def discover_samples(data_dir: Path) -> tuple[list[Sample], dict[str, object]]:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'data directory does not exist: {data_dir}')
    sample_root = locate_sample_root(data_dir)
    manifest = locate_manifest(data_dir)
    if manifest is not None:
        with manifest.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or 'path' not in reader.fieldnames:
                raise ValueError(f'manifest must contain a path column: {manifest}')
            names = [normalized_manifest_name(row['path']) for row in reader]
        source = 'test.csv'
    else:
        names = sorted((path.name for path in sample_root.iterdir() if path.is_dir()))
        source = 'sorted_sample_directories'
    if not names or len(names) != len(set(names)):
        raise ValueError('sample list is empty or contains duplicate sample identifiers')
    samples: list[Sample] = []
    for name in names:
        directory = sample_root / name
        if not directory.is_dir():
            raise FileNotFoundError(f'manifest sample directory is missing: {directory}')
        samples.append(Sample(f'small_model_track_test/{name}/', directory))
    return (samples, {'order_source': source, 'manifest': str(manifest) if manifest else None, 'sample_root': str(sample_root)})

def temporal_segment_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0 or sample_count <= 0:
        raise ValueError('frame_count and sample_count must be positive')
    indices: list[int] = []
    for segment in range(sample_count):
        left = segment * frame_count / sample_count
        right = (segment + 1) * frame_count / sample_count
        indices.append(min(frame_count - 1, int((left + right) / 2.0)))
    return indices

def sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob('*.png'), key=lambda value: value.name) if path.is_dir() else []

def load_video(path: Path, transform: CenterCropVideoTransform) -> tuple[torch.Tensor | None, int, int]:
    frames = sorted_pngs(path)
    if not frames:
        return (None, 0, 0)
    indices = temporal_segment_indices(len(frames), transform.frames)
    decoded: list[torch.Tensor] = []
    cache: dict[int, torch.Tensor | None] = {}
    invalid_files: set[int] = set()
    substitutions = 0
    for target_index in indices:
        selected_index: int | None = None
        value: torch.Tensor | None = None
        candidate_indices = sorted(range(len(frames)), key=lambda index: (abs(index - target_index), index))
        for candidate_index in candidate_indices:
            if candidate_index not in cache:
                try:
                    with Image.open(frames[candidate_index]) as image:
                        cache[candidate_index] = pil_to_tensor(image.convert('RGB'))
                except (OSError, ValueError, UnidentifiedImageError):
                    cache[candidate_index] = None
                    invalid_files.add(candidate_index)
            if cache[candidate_index] is not None:
                selected_index = candidate_index
                value = cache[candidate_index]
                break
        if selected_index is None or value is None:
            return (None, substitutions, len(invalid_files))
        substitutions += int(selected_index != target_index)
        decoded.append(value)
    transformed = transform(torch.stack(decoded))
    expected = (3, transform.frames, *transform.crop_size)
    if tuple(transformed.shape) != expected:
        raise ValueError(f'unexpected transformed video shape {tuple(transformed.shape)} for {path}')
    return (transformed.contiguous(), substitutions, len(invalid_files))

def write_submission(path: Path, rows: Iterable[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['path', 'prediction'])
        writer.writerows(rows)
    os.replace(temporary, path)
