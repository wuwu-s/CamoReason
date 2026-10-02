"""Per-image S-measure diff between two CamoReason prediction sets.

Usage:
    python scripts/per_image_diff.py \
        --a save/results_camoreason_test --b save/results_ab_no_text_freq \
        --out /tmp/per_image_diff.csv

Loads pred/gt PNGs from both result dirs, computes per-image Sm with the
repo's sod_metric (same implementation as eval), writes a CSV and prints
summary + top cases on each side.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sod_metric  # noqa: E402


def sm_one(pred, gt):
    m = sod_metric.Smeasure()
    m.step(pred=pred, gt=gt)
    return m.get_results()['sm']


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float64)  # 0..255, same scale as calc_cod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='result dir A (e.g. full model)')
    ap.add_argument('--b', required=True, help='result dir B (e.g. single-modality)')
    ap.add_argument('--out', default='/tmp/per_image_diff.csv')
    args = ap.parse_args()

    gt_dir = os.path.join(args.a, 'gt')
    pred_a = os.path.join(args.a, 'pred')
    pred_b = os.path.join(args.b, 'pred')
    stems = sorted(
        {f[:-4] for f in os.listdir(pred_a) if f.endswith('.png')}
        & {f[:-4] for f in os.listdir(pred_b) if f.endswith('.png')}
    )
    print(f'common images: {len(stems)} '
          f'(A-only: {len(os.listdir(pred_a)) - len(stems)}, '
          f'B-only: {len(os.listdir(pred_b)) - len(stems)})')

    rows = []
    for s in stems:
        gt = load_gray(os.path.join(gt_dir, s + '.png'))
        a = load_gray(os.path.join(pred_a, s + '.png'))
        b = load_gray(os.path.join(pred_b, s + '.png'))
        sm_a = sm_one(a, gt)
        sm_b = sm_one(b, gt)
        rows.append((s, sm_a, sm_b, sm_a - sm_b))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'sm_a', 'sm_b', 'diff'])
        w.writerows(rows)

    diffs = np.array([r[3] for r in rows])
    sm_a = np.array([r[1] for r in rows])
    sm_b = np.array([r[2] for r in rows])
    print(f'mean Sm A={sm_a.mean():.4f}  B={sm_b.mean():.4f}  '
          f'mean diff={diffs.mean():+.4f}  std={diffs.std():.4f}')
    print(f'A wins: {(diffs > 1e-4).sum()}  B wins: {(diffs < -1e-4).sum()}  '
          f'tie: {((diffs >= -1e-4) & (diffs <= 1e-4)).sum()}')
    print('\ntop-10 A > B (reasoning helps):')
    for s, a, b, d in sorted(rows, key=lambda r: -r[3])[:10]:
        print(f'  {s}  A={a:.4f} B={b:.4f} diff={d:+.4f}')
    print('\ntop-10 B > A (single-modality helps):')
    for s, a, b, d in sorted(rows, key=lambda r: r[3])[:10]:
        print(f'  {s}  A={a:.4f} B={b:.4f} diff={d:+.4f}')


if __name__ == '__main__':
    main()
