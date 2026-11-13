"""Prepare a portable comparison of the released ASCII models and full bundles."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kilix_bench.braille_reference import Packed
from kilix_bench.fonts import digest, revision


def prepare(directory, runtime):
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    weights = runtime / 'weights'
    record = weights / 'neuronte-ascii-next.validation.json'
    evidence = json.loads(record.read_text())
    if evidence['schema'] != 'neuronte.ascii-next-validation/v1' or evidence['seed'] != 0:
        raise ValueError('expected the released seed-0 ASCII validation record')
    checked = {}
    for line in (weights / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split(maxsplit=1)
        checked[name.lstrip('*')] = value
    names = ['neuronte-ascii-v0.nrt', 'neuronte-split-v5.nrt',
             'neuronte-ascii-groups-v1.nrt', 'neuronte-split-ascii-groups-v1.nrt',
             'neuronte-ascii-spatial-v1.nrt', 'neuronte-split-ascii-spatial-v1.nrt',
             record.name]
    for name in names:
        actual = digest(weights / name)
        if actual != checked.get(name) or (name in evidence['files'] and actual != evidence['files'][name]):
            raise ValueError('release SHA256 mismatch: ' + name)
    # Refuse to overwrite an earlier experiment, its targets, or its results.
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(runtime / 'train/decompose.py'), 'prepare', str(directory)],
                   check=True, capture_output=True, text=True)
    teacher = json.loads((directory / 'teacher.json').read_text())
    if digest(directory / 'teacher.npy') != evidence['teacher_sha256']:
        raise ValueError('the rebuilt teacher differs from the qualified release')
    corpus = list(range(32, 127))
    groups = [i for i in range(4) for _ in range(95*i//4, 95*(i+1)//4)]
    specifications = [
        ('ascii128', names[0], [0]*95, 'original-direct', 44801),
        ('split-v5', names[1], [0]*95, 'original-full', 44801),
        ('groups-direct', names[2], groups, 'repertoire-direct', 56836),
        ('groups-full', names[3], [g+6 for g in groups], 'repertoire-full', 56836),
        ('spatial-direct', names[4], [0]*95, 'spatial-direct', 28418),
        ('spatial-full', names[5], [6]*95, 'spatial-full', 28418),
    ]
    models, cases = {}, []
    for name, filename, routes, design, parameters in specifications:
        destination = directory / filename
        shutil.copyfile(weights / filename, destination)
        if digest(destination) != checked[filename]:
            raise ValueError('release changed during preparation: ' + filename)
        packed = Packed(destination.read_bytes())
        if [packed.route(c) for c in corpus] != routes:
            raise ValueError('release route differs: ' + filename)
        models[name] = {'file': filename, 'sha256': digest(destination),
                        'bytes': destination.stat().st_size}
        case = {'name': name, 'kind': 'bundle', 'members': [name], 'routes': routes,
                'design': design, 'stored_ascii_parameters': parameters,
                'canonical_patterns_per_handle': [95]}
        if not design.startswith('original-'):
            case['seed'] = 0
        cases.append(case)
    shutil.copyfile(record, directory / record.name)
    if digest(directory / record.name) != checked[record.name]:
        raise ValueError('validation record changed during preparation')
    manifest = {'schema': 'kilix-bench.ascii-experiment/v1', 'corpus': corpus,
                'models': models, 'cases': cases, 'targets': teacher['targets'],
                'teacher': teacher, 'runtime_revision': revision(runtime),
                'cohort': 'released-ascii-seed0',
                'qualification_note': 'This compares packaged seed 0, not the complete two-seed selection campaign.',
                'validation_record': {'file': record.name, 'sha256': checked[record.name]},
                'preparer_sha256': digest(Path(__file__))}
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='new directory on active storage')
    parser.add_argument('--neuronte-home', required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.directory, args.neuronte_home))


if __name__ == '__main__':
    main()
