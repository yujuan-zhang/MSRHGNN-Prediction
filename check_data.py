"""Check benchmark CSV formats without importing graph libraries or training."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COUNTS = {'B-dataset': (269, 598, 1021), 'C-dataset': (663, 409, 993),
          'F-dataset': (592, 313, 2741)}
MATRICES = {
    'drug': ['DrugFingerprint.csv', 'DrugGIP.csv', 'drug_ie_sim.csv'],
    'disease': ['DiseasePS.csv', 'DiseaseGIP.csv', 'disease_ie_sim.csv'],
    'protein': ['Protein_sequence.csv', 'ProteinGIP_Drug.csv', 'ProteinGIP_Disease.csv'],
}
FEATURES = {'drug': 'drug_f512_feature.csv', 'disease': 'disease_f512_feature.csv',
            'protein': 'protein_f512_feature.csv'}
EDGES = {'DrugDiseaseAssociationNumber.csv': ('drug', 'disease'),
         'DrugProteinAssociationNumber.csv': ('drug', 'protein'),
         'ProteinDiseaseAssociationNumber.csv': ('disease', 'protein')}


def matrix_shape(path, size, indexed):
    count = 0
    with path.open(newline='') as handle:
        reader = csv.reader(handle)
        if indexed:
            header = next(reader, [])
            if len(header) != size + 1:
                raise ValueError(f'{path.name}: expected a header with {size + 1} columns.')
        for line, row in enumerate(reader, 2 if indexed else 1):
            if len(row) != size + int(indexed):
                raise ValueError(f'{path.name}:{line}: wrong column count.')
            try:
                valid = all(math.isfinite(float(value)) for value in row[int(indexed):])
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f'{path.name}:{line}: cells must be finite numbers.')
            count += 1
    if count != size:
        raise ValueError(f'{path.name}: expected {size} rows, found {count}.')
    return [count, size]


def edge_rows(path, limits):
    count = 0
    with path.open(newline='') as handle:
        reader = csv.reader(handle)
        if len(next(reader, [])) != 2:
            raise ValueError(f'{path.name}: expected a two-column header.')
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f'{path.name}:{line}: expected two indices.')
            try:
                indices = [int(value) for value in row]
            except ValueError:
                raise ValueError(f'{path.name}:{line}: indices must be integers.') from None
            if not all(0 <= index < limit for index, limit in zip(indices, limits)):
                raise ValueError(f'{path.name}:{line}: index is outside entity range {limits}.')
            count += 1
    if not count:
        raise ValueError(f'{path.name}: no associations.')
    return count


def check(folder, dataset):
    required = [name for files in MATRICES.values() for name in files] + list(FEATURES.values()) + list(EDGES)
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ValueError('Missing files in ' + str(folder) + ': ' + ', '.join(missing))
    counts = dict(zip(['drug', 'disease', 'protein'], COUNTS[dataset]))
    shapes = {}
    for entity, names in MATRICES.items():
        for name in names:
            shapes[name] = matrix_shape(folder / name, counts[entity], indexed=True)
    for entity, name in FEATURES.items():
        shapes[name] = matrix_shape(folder / name, counts[entity], indexed=False)
    edges = {name: edge_rows(folder / name, [counts[entity] for entity in entities])
             for name, entities in EDGES.items()}
    return {'dataset': dataset, 'entities': counts, 'matrix_shapes': shapes,
            'association_rows': edges, 'files_checked': len(required),
            'scope': 'CSV formats and index ranges only; training not executed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=COUNTS, default='B-dataset')
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    folder = args.data_dir if args.data_dir else ROOT / 'data' / args.dataset
    try:
        report = check(folder, args.dataset)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f'ERROR: {error}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
