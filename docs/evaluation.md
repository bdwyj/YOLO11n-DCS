# Evaluation

The manuscript evaluates one-class horizontal ship boxes with IoU thresholds
0.50:0.05:0.95. Each retained ground-truth box matches at most one prediction at
each IoU threshold. Duplicate predictions and predictions without a valid match
are false positives. Unmatched valid ground truth is a false negative.

For DOTA, visible fragments with retained polygon area below 70% have no positive
label. The fragment CSV records their geometry. Evaluation uses the retained
ground-truth boxes without fragment ignore regions: predictions on unlabeled
fragments count as false positives when unmatched; undetected unlabeled fragments
do not create false negatives. Tiles with no retained positive label are excluded.

All comparisons use the saved train/validation/test lists. Training starts from
random initialization; validation selects best.pt; detection metrics are computed
on the test set. NPU timing and model-conversion accuracy are separate measurements.

## Evaluator and execution

The reference implementation is the horizontal-box detection validator in the
matching Ultralytics project, version 8.3.163 for the repeated experiments.
`tools/evaluate.py` is a release-time execution wrapper around that validator.
It requires an explicit JSON protocol copied from the actual evaluation settings:
`imgsz`, `batch`, `conf`, `iou` (NMS threshold), `max_det`, `rect`, `half`,
`device`, `workers`, `augment`, and `agnostic_nms`. The NMS IoU threshold and
the AP matching IoU thresholds serve different purposes.

```bash
python -m pip install -r requirements-evaluation.txt
python tools/evaluate.py --weights /path/to/best.pt --data /path/to/data.yaml --protocol /path/to/recorded_protocol.json --project /new/evaluation
```

Use the matching training project's custom model definitions when evaluating
DCS checkpoints; `--register-module` can import its registration module.
The wrapper records explicit settings, software version and returned metrics.
The release contains dataset preparation and evaluation support; pretrained
checkpoints and a complete custom-model training environment are separate artifacts.

Upstream source: https://github.com/ultralytics/ultralytics/tree/v8.3.163/ultralytics/models/yolo/detect
