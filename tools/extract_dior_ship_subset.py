"""Reconstruct ship-subset extraction from the author's stated processing rule.

This newly written script is not the recovered historical training script.
Keep complete images containing ships, retain ship annotations only, and export
single-class YOLO labels. Images are copied without cropping or resizing.
Optionally replay existing train/val/test ID lists; no random split is inferred.
Python standard library only.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SPLITS = ('train', 'val', 'test')


def index_files(directory, extensions):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in result:
                raise ValueError(f'Duplicate image/annotation ID: {path.stem}')
            result[path.stem] = path
    return result


def read_xml_ship_labels(path, ship_name):
    root = ET.parse(path).getroot()
    objects = [obj for obj in root.findall('object')
               if (obj.findtext('name') or '').strip().casefold() == ship_name.casefold()]
    if not objects:
        return []
    width = float(root.findtext('size/width'))
    height = float(root.findtext('size/height'))
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image dimensions: {path}')
    lines = []
    for obj in objects:
        box = obj.find('bndbox')
        if box is None:
            raise ValueError(f'Missing horizontal bndbox for ship: {path}')
        xmin, ymin, xmax, ymax = [float(box.findtext(key))
                                for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        values = ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                  (xmax - xmin) / width, (ymax - ymin) / height)
        lines.append('0 ' + ' '.join(f'{value:.10g}' for value in values))
    return lines


def read_yolo_ship_labels(path, ship_class_id):
    lines = []
    for number, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = text.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'Expected horizontal YOLO box at {path}:{number}')
        if int(fields[0]) == ship_class_id:
            # Preserve existing coordinate strings, including any original defects.
            lines.append('0 ' + ' '.join(fields[1:]))
    return lines


def read_existing_splits(directory):
    assignments = {}
    for split in SPLITS:
        path = directory / f'{split}_image_ids.txt'
        for text in path.read_text(encoding='utf-8-sig').splitlines():
            image_id = text.strip()
            if not image_id:
                continue
            if image_id in assignments:
                raise ValueError(f'Duplicate ID within/across split lists: {image_id}')
            assignments[image_id] = split
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--format', choices=('xml', 'yolo'), default='xml')
    parser.add_argument('--ship-name', default='ship', help='XML object name, case-insensitive')
    parser.add_argument('--ship-class-id', type=int, help='Required for YOLO input labels')
    parser.add_argument('--split-records', type=Path, help='Folder with three existing *_image_ids.txt lists')
    parser.add_argument('--output', type=Path, help='A new output directory')
    parser.add_argument('--dry-run', action='store_true', help='Read and summarize without writing')
    args = parser.parse_args()
    if not args.images.is_dir() or not args.annotations.is_dir():
        parser.error('Image and annotation directories must exist.')
    if args.format == 'yolo' and args.ship_class_id is None:
        parser.error('--ship-class-id must identify ships in the input YOLO dataset.')
    if not args.dry_run and args.output is None:
        parser.error('Provide --output or use --dry-run.')
    if args.output is not None and args.output.exists() and not args.dry_run:
        parser.error('Output already exists; choose a new directory.')

    images = index_files(args.images, IMAGE_EXTENSIONS)
    annotations = index_files(args.annotations, {'.xml' if args.format == 'xml' else '.txt'})
    missing = sorted(set(images) - set(annotations))
    if missing:
        raise ValueError(f'{len(missing)} images lack annotation files; examples: {missing[:5]}')
    selected = {}
    for image_id in sorted(images):
        label = annotations[image_id]
        lines = (read_xml_ship_labels(label, args.ship_name) if args.format == 'xml'
                 else read_yolo_ship_labels(label, args.ship_class_id))
        if lines:
            selected[image_id] = lines
    assignments = read_existing_splits(args.split_records) if args.split_records else {}
    if args.split_records and set(assignments) != set(selected):
        raise ValueError('Existing split IDs must exactly cover the selected ship images. '
                         f'Missing from lists: {len(set(selected)-set(assignments))}; '
                         f'not selected: {len(set(assignments)-set(selected))}.')

    summary = dict(script_provenance='New reconstruction from author-described selection rule',
                   input_format=args.format, input_images=len(images),
                   selected_images=len(selected), ship_instances=sum(map(len, selected.values())),
                   excluded_images_without_ship=len(images)-len(selected),
                   image_processing='Copy full images unchanged; no cropping, tiling, or resizing',
                   split_method='Replay provided IDs' if args.split_records else 'No split performed')
    if assignments:
        summary['splits'] = {s:sum(value == s for value in assignments.values()) for s in SPLITS}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for image_id, lines in selected.items():
        split = assignments.get(image_id, '')
        image_folder = args.output/'images'/split
        label_folder = args.output/'labels'/split
        image_folder.mkdir(parents=True, exist_ok=True)
        label_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[image_id], image_folder/images[image_id].name)
        (label_folder/f'{image_id}.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        rows.append(dict(image_id=image_id, split=split, ship_instances=len(lines),
                         source_image=str(images[image_id]), source_annotation=str(annotations[image_id])))
    records = args.output/'split_records'
    records.mkdir()
    with (records/'selection_manifest.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('image_id','split','ship_instances','source_image','source_annotation'))
        writer.writeheader()
        writer.writerows(rows)
    (records/'selected_image_ids.txt').write_text(''.join(i+'\n' for i in selected), encoding='utf-8')
    if assignments:
        for split in SPLITS:
            (records/f'{split}_image_ids.txt').write_text(
                ''.join(i+'\n' for i in selected if assignments[i] == split), encoding='utf-8')
        (args.output/'data.yaml').write_text(
            f'path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n', encoding='utf-8')
    (records/'selection_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
