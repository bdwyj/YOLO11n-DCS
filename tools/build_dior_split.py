"""Recorded seed-42 DIOR grouping and allocation algorithm. Supply the experiment input image/label version."""
import argparse, collections, csv, hashlib, json, math, random, shutil, sys
from pathlib import Path
SPLITS=('train','val','test')
IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
LABEL_COORDINATE_ISSUES=[]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def labels_info(path):
    counts = collections.Counter()
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != '0':
            raise ValueError(f'Invalid ship YOLO label: {path}:{lineno}: {line}')
        box = [float(x) for x in parts[1:]]
        if not all(math.isfinite(x) for x in box):
            raise ValueError(f'Nonfinite box coordinates: {path}:{lineno}: {line}')
        if not all(0 <= x <= 1 for x in box) or min(box[2:]) <= 0:
            LABEL_COORDINATE_ISSUES.append(dict(label_path=str(path), image_id=path.stem,
                line_number=lineno, original_line=line,
                reason='Nonpositive width/height or coordinates outside [0,1]',
                action='Preserved unchanged; dataset splitting does not clean annotations'))
        counts['0'] += 1
    return counts['0']

def indexed_files(folder, extensions):
    result = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in extensions:
            if p.stem in result:
                raise ValueError(f'Duplicate image ID: {p.stem}')
            result[p.stem] = p
    return result

def export_records(folder, rows):
    folder.mkdir(parents=True, exist_ok=False)
    write_csv(folder / 'manifest.csv', rows)
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        (folder / f'{split}_image_ids.txt').write_text(
            ''.join(r['image_id'] + '\n' for r in selected), encoding='utf-8')
        (folder / f'{split}_images.txt').write_text(
            ''.join(r['image_path'] + '\n' for r in selected), encoding='utf-8')
        if 'source_id' in rows[0]:
            ids = sorted({r['source_id'] for r in selected})
            (folder / f'{split}_source_ids.txt').write_text(
                ''.join(x + '\n' for x in ids), encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('replay_split_local_archive.py'), folder / 'replay_split.py')

def select_groups_exact(groups, target):
    """Select first reachable exact total in seeded group order, without splitting a group."""
    parents = [None] * (target + 1)
    parents[0] = (-1, -1)
    for index, group in enumerate(groups):
        size = len(group)
        for total in range(target, size - 1, -1):
            if parents[total] is None and parents[total-size] is not None:
                parents[total] = (total-size, index)
        if parents[target] is not None:
            break
    if parents[target] is None:
        raise ValueError(f'Cannot meet exact split count {target} without separating duplicate groups')
    selected = set()
    total = target
    while total:
        total, index = parents[total]
        selected.add(index)
    return ([g for i,g in enumerate(groups) if i in selected],
            [g for i,g in enumerate(groups) if i not in selected])

def prepare_dior(source, target, seed):
    LABEL_COORDINATE_ISSUES.clear()
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    images = indexed_files(source / 'images', IMAGE_EXTS)
    labels = indexed_files(source / 'labels', {'.txt'})
    missing = sorted(set(images) - set(labels))
    orphan = sorted(set(labels) - set(images))
    if missing:
        raise ValueError(f'Missing labels: {missing}')
    ids = sorted(images)
    ntrain = len(ids) * 7 // 10
    nval = len(ids) * 2 // 10
    rows = []
    image_hash_ids = collections.defaultdict(list)
    # Audit everything before creating any output dataset.
    records_by_id = {}
    for image_id in ids:
        img, lab = images[image_id], labels[image_id]
        ih, lh = digest(img), digest(lab)
        image_hash_ids[ih].append(image_id)
        records_by_id[image_id] = dict(split='', image_id=image_id, image_file=img.name,
            image_path='', label_path='', original_image=str(img), original_label=str(lab),
            ship_instances=labels_info(lab), image_sha256=ih, label_sha256=lh)
    duplicates = [v for v in image_hash_ids.values() if len(v) > 1]
    groups = sorted(image_hash_ids.values(), key=lambda g:g[0])
    random.Random(seed).shuffle(groups)
    train_groups, rest = select_groups_exact(groups, ntrain)
    val_groups, test_groups = select_groups_exact(rest, nval)
    assignments = {s:[i for group in grouped for i in group]
                   for s,grouped in zip(SPLITS,(train_groups,val_groups,test_groups))}
    for split in SPLITS:
        for image_id in assignments[split]:
            row = records_by_id[image_id]
            row['split'] = split
            row['image_path'] = str(target / 'images' / split / images[image_id].name)
            row['label_path'] = str(target / 'labels' / split / labels[image_id].name)
            rows.append(row)
    duplicate_details = [dict(image_ids=g, split=records_by_id[g[0]]['split'],
        identical_label_files=len({records_by_id[i]['label_sha256'] for i in g})==1) for g in duplicates]
    for split in SPLITS:
        (target / 'images' / split).mkdir(parents=True)
        (target / 'labels' / split).mkdir(parents=True)
    for i, row in enumerate(rows, 1):
        for kind in ('image', 'label'):
            original, dest = Path(row['original_' + kind]), Path(row[kind + '_path'])
            shutil.copy2(original, dest)
            if digest(dest) != row[kind + '_sha256']:
                raise ValueError(f'Copy verification failed: {dest}')
        if i % 500 == 0:
            print(f'DIOR copied and hash-verified {i}/{len(rows)} pairs', flush=True)
    records = target / 'split_records'
    export_records(records, rows)
    (records / 'input_image_ids_sorted.txt').write_text(
        ''.join(i + '\n' for i in sorted(images)), encoding='utf-8')
    write_json(records / 'excluded_unpaired_labels.json', [
        dict(label_id=i, path=str(labels[i]), reason='No corresponding image; excluded from splitting',
             content=labels[i].read_text(encoding='utf-8-sig')) for i in orphan])
    write_json(records / 'label_coordinate_issues.json', LABEL_COORDINATE_ISSUES)
    write_json(records / 'duplicate_image_groups.json', duplicate_details)
    summary = dict(dataset='DIOR-Ship', seed=seed, seed_provenance='Chosen explicitly by this script',
        source=str(source), target=str(target), python_version=sys.version,
        method='Sort image IDs lexicographically; group byte-identical images by SHA-256; '
               'sort groups by first image ID; shuffle groups once using random.Random(seed).shuffle; '
               'use select_groups_exact descending dynamic programming in group order to select '
               'floor(0.7*N) train images, then floor(0.2*N) val images; remainder test',
        split_unit='SHA-256 group of identical images; all original image IDs retained', requested_ratios=[0.7, 0.2, 0.1],
        total_images=len(images), total_ship_instances=sum(r['ship_instances'] for r in rows),
        excluded_unpaired_labels=orphan, exact_duplicate_image_groups=duplicates,
        exact_duplicate_groups_with_different_labels=sum(not g['identical_label_files'] for g in duplicate_details),
        cross_split_identical_image_sha256_count=0,
        label_coordinate_issue_count=len(LABEL_COORDINATE_ISSUES),
        label_coordinate_issue_image_count=len({r['image_id'] for r in LABEL_COORDINATE_ISSUES}),
        annotation_policy='Original labels preserved byte-for-byte, including recorded coordinate issues',
        original_files_modified=False, file_creation='Independent copies; SHA-256 verified',
        splits={s:dict(image_count=len(assignments[s]),
                       ship_instances=sum(r['ship_instances'] for r in rows if r['split']==s),
                       actual_fraction=len(assignments[s])/len(images)) for s in SPLITS},
        cross_split_image_id_intersections=0,
        manifest_sha256=digest(records / 'manifest.csv'))
    write_json(records / 'summary.json', summary)
    (target / 'data.yaml').write_text(
        'path: ' + target.as_posix() + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n',
        encoding='utf-8')
    (records / 'README_划分说明.md').write_text(
        '# DIOR 船舶数据集划分记录\n\n'
        f'随机种子：{seed}。目标比例 train:val:test = 7:2:1。\n\n'
        '先将全部图像文件名去扩展名作为 ID，并按字符串升序排序；按图像 SHA-256 将内容完全相同的图像分组，'
        '组内 ID 升序，各组按第一个 ID 升序；使用 Python 标准库 '
        f'`random.Random({seed}).shuffle(groups)` 打乱组列表一次。按该组顺序执行 '
        '`select_groups_exact` 子集和动态规划（状态从目标值向下更新，首次达到目标即停止），'
        '先选择总数恰为 `N*7//10` 的完整组作为训练集，再从剩余组中选择总数恰为 `N*2//10` '
        '的完整组作为验证集，剩余组用于测试；整数取整造成轻微比例差异。'
        '未进行类别分层或按船舶实例数配平。\n\n'
        f'训练 {ntrain} 张，验证 {nval} 张，测试 {len(ids)-ntrain-nval} 张。\n\n'
        '`train_image_ids.txt`、`val_image_ids.txt`、`test_image_ids.txt` 保留前导零，'
        '每行一个 ID，顺序为随机组列表中所选组的顺序，组内 ID 升序。`manifest.csv` 记录源路径、目标路径、'
        '船舶数及图像和标签 SHA-256。原始图像与标签未修改，输出为独立副本。\n\n'
        '输入中没有对应图像的标签见 `excluded_unpaired_labels.json`，不会作为样本参与划分。'
        '所有样本均检查图像与标签配对、YOLO 单类标注及 ID 无交集。'
        '21 组内容重复的图像各自保留原 ID，整组放入同一子集，避免相同图像跨集合泄漏；'
        '详情（包括重复图像的标签文件是否相同）见 `duplicate_image_groups.json`。'
        '坐标异常记录在 `label_coordinate_issues.json`，原标签逐字节保留；本次只划分，不清洗标注。'
        '本检查不判断不同图像是否覆盖相近的地理区域。\n\n'
        '精确复现现有划分（无需依赖随机数生成器版本）：\n\n'
        '```powershell\npython replay_split.py --manifest manifest.csv --output /new/dior_replayed\n```\n\n'
        '重新由种子生成：`python tools/build_dior_split.py --source /path/to/dior_ship_inputs --output /new/dior_regenerated --seed 42`；'
        '输入 ID 和图像内容必须相同。\n',
        encoding='utf-8')
    shutil.copy2(__file__, records / Path(__file__).name)
    print('DIOR_SUMMARY ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Folder containing images/ and labels/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    prepare_dior(a.source.resolve(),a.output.resolve(),a.seed)
