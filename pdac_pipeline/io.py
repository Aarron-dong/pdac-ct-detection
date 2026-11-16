from __future__ import annotations
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import SimpleITK as sitk

def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def read_image(path: Path):
    if str(path).isascii():
        return sitk.ReadImage(str(path), sitk.sitkFloat32)
    with tempfile.TemporaryDirectory(prefix='pdac-image-') as staging:
        staged = Path(staging) / 'image.nii.gz'
        shutil.copyfile(path, staged)
        return sitk.ReadImage(str(staged), sitk.sitkFloat32)

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temporary, path)

def bundle_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle members must be relative paths inside the package.')
    return path

def write_image(array, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    temporary = path.with_name(path.name.replace('.nii.gz', '.writing.nii.gz'))
    if str(temporary).isascii():
        sitk.WriteImage(image, str(temporary), True)
    else:
        with tempfile.TemporaryDirectory(prefix='pdac-image-') as staging:
            staged = Path(staging) / 'image.nii.gz'
            sitk.WriteImage(image, str(staged), True)
            shutil.copyfile(staged, temporary)
    os.replace(temporary, path)
