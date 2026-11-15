r"""Create DOTA Ship image tiles and YOLO horizontal-box labels.

Example:
  python build_dota_ship_tiles.py --images RAW/images \
      --annotations RAW/labelTxt --output NEW/output

The sliding window is 1024 pixels with 200 pixels of overlap. Edge crops keep
their actual width and height. Empty tiles are omitted unless --keep-empty is
given. A CSV records subthreshold fragments visible in retained tiles.
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


WINDOW = 1024
OVERLAP = 200
STRIDE = WINDOW - OVERLAP
AREA_THRESHOLD = 0.7


def area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in
                   zip(poly, poly[1:] + poly[:1]))) / 2.0


def clip_edge(poly, axis, boundary, keep_greater):
    if not poly:
        return []

    def inside(point):
        value = point[axis]
        return value >= boundary if keep_greater else value <= boundary

    def intersection(a, b):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            return a
        t = (boundary - a[axis]) / delta
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    output = []
    previous = poly[-1]
    for current in poly:
        if inside(current):
            if not inside(previous):
                output.append(intersection(previous, current))
            output.append(current)
        elif inside(previous):
            output.append(intersection(previous, current))
        previous = current
    return output


def clip_polygon(poly, x0, y0, x1, y1):
    for axis, boundary, keep_greater in (
        (0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)
    ):
        poly = clip_edge(poly, axis, boundary, keep_greater)
    return poly


def read_dota_ship_labels(path):
    objects = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.strip().split()
        if len(fields) < 9:
            continue  # DOTA headers such as imagesource/gsd
        if fields[8].lower() != "ship":
            continue
        try:
            numbers = [float(value) for value in fields[:8]]
        except ValueError as exc:
            raise ValueError(f"Bad polygon at {path}:{line_no}") from exc
        poly = [(numbers[i], numbers[i + 1]) for i in range(0, 8, 2)]
        if area(poly) > 0:
            objects.append((line_no, poly))
    return objects


def process_one(image_path, label_path, images_out, labels_out, ignored, keep_empty):
    objects = read_dota_ship_labels(label_path)
    with Image.open(image_path) as source:
        width, height = source.size
        for y0 in range(0, height, STRIDE):
            for x0 in range(0, width, STRIDE):
                x1, y1 = min(x0 + WINDOW, width), min(y0 + WINDOW, height)
                tile_w, tile_h = x1 - x0, y1 - y0
                stem = f"{image_path.stem}_{x0}_{y0}"
                yolo_rows = []
                ignored_rows = []

                for line_no, polygon in objects:
                    clipped = clip_polygon(polygon, x0, y0, x1, y1)
                    clipped_area = area(clipped)
                    if clipped_area <= 0:
                        continue
                    ratio = clipped_area / area(polygon)
                    local = [(x - x0, y - y0) for x, y in clipped]
                    if ratio < AREA_THRESHOLD:
                        ignored_rows.append((stem, image_path.stem, line_no, ratio,
                                             json.dumps(local)))
                        continue
                    left = min(p[0] for p in local)
                    right = max(p[0] for p in local)
                    top = min(p[1] for p in local)
                    bottom = max(p[1] for p in local)
                    if right <= left or bottom <= top:
                        continue
                    xc = (left + right) / (2 * tile_w)
                    yc = (top + bottom) / (2 * tile_h)
                    bw = (right - left) / tile_w
                    bh = (bottom - top) / tile_h
                    yolo_rows.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")

                if not yolo_rows and not keep_empty:
                    continue
                source.crop((x0, y0, x1, y1)).save(images_out / f"{stem}.png")
                (labels_out / f"{stem}.txt").write_text(
                    "\n".join(yolo_rows) + ("\n" if yolo_rows else ""), encoding="utf-8"
                )
                ignored.extend(ignored_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-empty", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in
                    {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not images:
        parser.error("No images found")
    images_out = args.output / "images"
    labels_out = args.output / "labels"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    ignored = []
    for image in images:
        label = args.annotations / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        process_one(image, label, images_out, labels_out, ignored, args.keep_empty)
    with (args.output / "ignored_fragments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_id", "source_image_id", "annotation_line", "retained_area_ratio",
                         "clipped_polygon_local_xy"])
        writer.writerows(ignored)
    print(f"Wrote {len(list(images_out.iterdir()))} tiles and {len(ignored)} ignored fragments")


if __name__ == "__main__":
    main()
