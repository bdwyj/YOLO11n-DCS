"""Reconstruct the archived DOTA allocation from a portable manifest.

This replays saved assignments; it does not regenerate the original tiling or
seeded grouping procedure. It never overwrites an existing output directory.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,shutil

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--manifest',type=Path,required=True)
 ap.add_argument('--images-root',type=Path,required=True)
 ap.add_argument('--labels-root',type=Path,required=True)
 ap.add_argument('--output',type=Path)
 ap.add_argument('--check-only',action='store_true')
 ap.add_argument('--limit-per-split',type=int,help='Validate a limited sample; allowed only with --check-only')
 a=ap.parse_args()
 if not a.check_only and (a.output is None or a.output.exists()):
  ap.error('Reconstruction requires --output naming a directory that does not exist.')
 if a.limit_per_split and not a.check_only:ap.error('--limit-per-split requires --check-only.')
 if a.limit_per_split is not None and a.limit_per_split<1:ap.error('The limit must be positive.')
 with a.manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 indexes=[]
 for root in [a.images_root,a.labels_root]:
  index=defaultdict(list)
  for p in root.rglob('*'):
   if p.is_file():index[p.name].append(p)
  indexes.append(index)
 counts=defaultdict(int);operations=[];seen=set()
 for r in rows:
  split=r['split']
  if split not in ['train','val','test']:raise ValueError('Unknown split '+split)
  if a.limit_per_split and counts[split]>=a.limit_per_split:continue
  if (split,r['image_id']) in seen:raise ValueError('Duplicate manifest record')
  seen.add((split,r['image_id']))
  for kind,index in zip(['image','label'],indexes):
   name=r[kind+'_file'];expected=r[kind+'_sha256']
   matches=[p for p in index.get(name,[]) if digest(p)==expected]
   if not matches:raise ValueError('Missing file or hash mismatch: '+name)
   operations.append((matches[0],split,'images' if kind=='image' else 'labels',name,expected))
  counts[split]+=1
 if not a.check_only:
  for src,split,folder,name,expected in operations:
   dst=a.output/folder/split/name;dst.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(src,dst)
   if digest(dst)!=expected:raise ValueError('Copied file hash mismatch: '+str(dst))
  shutil.copy2(a.manifest,a.output/'manifest.csv')
  (a.output/'data.yaml').write_text('path: '+a.output.resolve().as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n',encoding='utf-8')
 print(('Validated' if a.check_only else 'Reconstructed')+' '+str(dict(counts))+' image-label pairs with SHA-256 checks.')

if __name__=='__main__':main()
