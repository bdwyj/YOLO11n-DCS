"""Rebuild an exact split from an archived manifest; never overwrite output."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {target}')
    with args.manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    operations = []
    for row in rows:
        split = row['split']
        if split not in ('train','val','test'):
            raise ValueError(f'Invalid split: {split}')
        for kind, folder in (('image','images'),('label','labels')):
            # Prefer archived dataset files; fall back to the original inputs.
            source = Path(row[kind + '_path'])
            if not source.is_file():
                source = Path(row['original_' + kind])
            if not source.is_file() or sha256(source) != row[kind + '_sha256']:
                raise ValueError(f'Missing or changed input file: {source}')
            dest = target / folder / split / Path(row[kind + '_path']).name
            if dest in seen:
                raise ValueError(f'Duplicate output file: {dest}')
            seen.add(dest)
            operations.append((source, dest, row[kind + '_sha256']))
    for source, dest, expected in operations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if sha256(dest) != expected:
            raise ValueError(f'Copy verification failed: {dest}')
    records = target / 'split_records'
    records.mkdir()
    shutil.copy2(args.manifest, records / 'manifest.csv')
    for split in ('train','val','test'):
        (records / f'{split}_image_ids.txt').write_text(
            ''.join(r['image_id'] + '\n' for r in rows if r['split']==split), encoding='utf-8')
        if rows and 'source_id' in rows[0]:
            source_ids = sorted({r['source_id'] for r in rows if r['split']==split})
            (records / f'{split}_source_ids.txt').write_text(
                ''.join(i + '\n' for i in source_ids), encoding='utf-8')
    (target / 'data.yaml').write_text(
        f'path: {target.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n',
        encoding='utf-8')
    print(f'Rebuilt {len(rows)} image-label pairs with SHA-256 verification: {target}')


if __name__ == '__main__':
    main()
