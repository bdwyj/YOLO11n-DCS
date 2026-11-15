"""Explicit test-set entry point for the Ultralytics horizontal-box evaluator.

Release helper: supply the settings saved for the experiment being reproduced.
Install custom model modules in the matching training project before loading DCS weights.
"""
import argparse, importlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True,help='JSON with explicit validation settings')
    p.add_argument('--project',type=Path,required=True,help='New evaluation output directory')
    p.add_argument('--register-module',action='append',default=[],help='Import custom model registration module')
    a=p.parse_args()
    required={'imgsz','batch','conf','iou','max_det','rect','half','device','workers','augment','agnostic_nms'}
    cfg=json.loads(a.protocol.read_text(encoding='utf-8'))
    missing=required-set(cfg)
    if missing: p.error('Missing protocol settings: '+', '.join(sorted(missing)))
    if set(cfg)-required: p.error('Unknown protocol settings: '+', '.join(sorted(set(cfg)-required)))
    if any(cfg[k] is None for k in required):p.error('Every protocol setting must have an explicit value.')
    if not a.weights.is_file() or not a.data.is_file():p.error('Weights and dataset YAML must exist.')
    if a.project.exists():p.error('--project must name a new output directory.')
    for module in a.register_module:importlib.import_module(module)
    import ultralytics
    from ultralytics import YOLO
    model=YOLO(str(a.weights),task='detect')
    metrics=model.val(data=str(a.data.resolve()),split='test',project=str(a.project.resolve()),
                      name='test',exist_ok=False,plots=False,save_json=False,**cfg)
    out=Path(metrics.save_dir)
    report={'ultralytics':ultralytics.__version__,'split':'test','weights':a.weights.name,
            'protocol':cfg,'metrics':{k:float(v) for k,v in metrics.results_dict.items()}}
    (out/'evaluation_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
