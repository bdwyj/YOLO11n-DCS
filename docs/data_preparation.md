# Dataset preparation

Original imagery and original annotations are obtained from the dataset providers:
- DOTA: https://captain-whu.github.io/DOTA/dataset.html
- DIOR: https://gcheng-nwpu.github.io/

## Replay the experimental allocations

The manifests record image and label SHA-256 values and portable filenames.
Provide image/label roots containing the experiment-version inputs. The replay
tool finds matching filenames and verifies hashes before copying files.

```bash
python tools/replay_split.py --manifest splits/dota/manifest.csv --images-root /path/to/dota/tiles/images --labels-root /path/to/dota/tiles/labels --check-only
python tools/replay_split.py --manifest splits/dota/manifest.csv --images-root /path/to/dota/tiles/images --labels-root /path/to/dota/tiles/labels --output /new/dota_ship
python tools/replay_split.py --manifest splits/dior/manifest.csv --images-root /path/to/dior/images --labels-root /path/to/dior/labels --output /new/dior_ship
```

The output uses `images/{train,val,test}`, `labels/{train,val,test}` and a
`data.yaml` file with ship as class 0. Existing outputs are not overwritten.

## DOTA tiles

```bash
python -m pip install -r requirements-data.txt
python tools/build_dota_ship_tiles.py --images /path/to/dota/images --annotations /path/to/dota/labelTxt --output /new/dota_tiles
```

The supplied construction script uses 1024-pixel windows, 200-pixel overlap and
actual border-tile dimensions. It retains ship polygons with clipped/original
area at least 0.7, converts them to horizontal boxes and saves positive tiles.
The fixed manifests define the exact experimental assignments. All tiles from
one source ID belong to one split. Replay the saved assignments after tiling.

## DIOR ships

Complete images containing ships are retained without cropping. Other class
labels are removed. `tools/extract_dior_ship_subset.py` reconstructs this selection
rule from XML or existing YOLO labels; its XML conversion uses declared image
dimensions. For exact experiment replay, use the image/label hashes in the manifest.
Four nonpositive boxes are present in the archived raw input label files; the
valid-box counts are 44,517 / 12,414 / 5,465. Source image and normalization versions
are identified by the published hashes.

```bash
python tools/extract_dior_ship_subset.py --images /path/to/dior/images --annotations /path/to/dior/labels --format yolo --ship-class-id 0 --output /new/dior_ship_inputs
python tools/build_dior_split.py --source /new/dior_ship_inputs --output /new/dior_seed42 --seed 42
```

The saved DIOR lists are the primary allocation records. The recorded allocation
algorithm sorts IDs, groups byte-identical images, shuffles groups with seed 42,
then selects whole groups to reach 1,891 training and 540 validation images;
the remaining 271 images form the test set. Use identical input image bytes
when regenerating assignments. Replaying the supplied lists preserves the
allocation across runtime versions.

## File provenance

`build_dota_ship_tiles.py` is the author-supplied DOTA construction script.
`build_dior_split.py` preserves the recorded DIOR allocation functions with
portable command-line input paths. `extract_dior_ship_subset.py` is a reconstruction
of the stated selection rule. `replay_split.py` and `evaluate.py` provide portable
release entry points. The SHA-256 index identifies the files in this release.
