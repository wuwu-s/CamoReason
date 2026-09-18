"""Forward-only NaN scan (TEMPORARY diagnostic; delete after root-cause fix).

Runs the training forward (bf16 autocast, with loss) over every sample of a
split at sam3.pt init weights, no backward. If any sample makes the forward
non-finite at init weights, the trigger is data-dependent (not trained-weights
dependent) and the sample id + first-broken loss component are printed.

Usage:
  python scripts/nan_scan.py --split val        # 890 samples, ~4 min
  python scripts/nan_scan.py --split train      # 3564 samples, ~15 min
  python scripts/nan_scan.py --split val --fp32 # same under fp32 (bf16 test)
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from camoreason.mhcd_dataset import MHCDDataset  # noqa: E402
from camoreason.model import CamoReason  # noqa: E402
import camoreason.train as camo_train  # noqa: E402 (reuse load_sam3_weights)


def comp_str(loss_log):
    return {k: float(v) for k, v in loss_log.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(REPO_ROOT, 'configs/camoreason-mhcd.yaml'))
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fp32', action='store_true', help='forward in fp32 instead of bf16')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    ds = MHCDDataset(**{**cfg[f'{args.split}_dataset']['args'], 'limit': args.limit})
    print(f'{args.split}: {len(ds)} samples')
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = CamoReason(**cfg['model']['args']).cuda().eval()
    camo_train.config = cfg  # load_sam3_weights reads the module-global
    camo_train.load_sam3_weights(model)

    n_bad = 0
    for bi, batch in enumerate(loader):
        fg = batch['gt'].flatten(1).sum(dim=1)
        empty = [str(sid) for sid, s in zip(batch['id'], fg) if s == 0]
        if empty:
            print(f'batch {bi}: EMPTY-GT sample(s): {empty}')

        with torch.no_grad(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=not args.fp32
        ):
            loss = model(batch['inp'].cuda(non_blocking=True),
                         batch['v'].cuda(non_blocking=True),
                         batch['t_pos'].cuda(non_blocking=True),
                         batch['t_neg'].cuda(non_blocking=True),
                         batch['gt'].cuda(non_blocking=True))
        if not torch.isfinite(loss):
            n_bad += 1
            print(f'[NON-FINITE] batch {bi}, ids={list(batch["id"])}, loss_log={comp_str(model.loss_log)}')
            # narrow to the single sample
            for j, sid in enumerate(batch['id']):
                one = {k: batch[k][j:j + 1].cuda() for k in ('inp', 'v', 't_pos', 't_neg', 'gt')}
                with torch.no_grad(), torch.autocast(
                    'cuda', dtype=torch.bfloat16, enabled=not args.fp32
                ):
                    loss1 = model(one['inp'], one['v'], one['t_pos'], one['t_neg'], one['gt'])
                tag = 'NON-FINITE' if not torch.isfinite(loss1) else 'ok'
                print(f'  sample {j} id={sid}: {tag} loss={float(loss1)} comp={comp_str(model.loss_log)}')
        if bi % 20 == 0:
            print(f'batch {bi}/{len(loader)}')
    print(f'done: {n_bad} non-finite batch(es)')


if __name__ == '__main__':
    main()
