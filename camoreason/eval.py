"""Evaluate a CamoReason checkpoint on the test split: Sm / Em / FwBeta / MAE.

Usage:
    python -m camoreason.eval --config configs/camoreason-mhcd.yaml \
        [--ckpt save/_camoreason-mhcd/model_epoch_best.pth] [--tag test]

Metrics follow the original repo protocol: metric_fn(sigmoid(infer(inp)), gt)
on the probability map at input resolution; per-image dumps (image/gt/pred)
are restored to the source resolution via batch['shape'] (test.py layout).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402

import utils  # noqa: E402
from camoreason.mhcd_dataset import MHCDDataset  # noqa: E402
from camoreason.model import CamoReason  # noqa: E402

IMG_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'configs/camoreason-mhcd.yaml'))
    parser.add_argument('--ckpt', default=None, help='checkpoint to evaluate (default: save/<name>/model_epoch_best.pth)')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--save-maps', action='store_true', help='dump image/gt/pred maps')
    parser.add_argument('--limit', type=int, default=None, help='evaluate only the first N images')
    parser.add_argument('--batch', type=int, default=None, help='override eval batch size')
    parser.add_argument('--out-dir', default=None, help='dump dir override (default: save/<name>)')
    parser.add_argument('--postproc', action='store_true',
                        help='saved preds: sigma2 gaussian -> 0.5 -> open2 -> close3 -> min-area 64 '
                             '(original pipeline; metrics still use the soft map)')
    parser.add_argument('--overlay', action='store_true',
                        help='also save the mask overlaid on the original image (light red tint)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_dir = os.path.join(REPO_ROOT, 'save', save_name)
    ck_path = args.ckpt or os.path.join(save_dir, 'model_epoch_best.pth')

    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model = CamoReason(**config['model']['args']).cuda()
    msg = model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"loaded {ck_path}")

    spec = dict(config['test_dataset'])
    spec['args'] = dict(spec['args'], split='test', limit=args.limit)
    loader = DataLoader(
        MHCDDataset(**spec['args']),
        batch_size=args.batch or spec.get('batch_size', 8),
        shuffle=False,
        num_workers=spec.get('num_workers', 8),
        pin_memory=True,
    )

    out_dir = args.out_dir or save_dir
    if args.save_maps:
        for p in ('image', 'gt', 'pred') + (('overlay',) if args.overlay else ()):
            os.makedirs(os.path.join(out_dir, p), exist_ok=True)
    sub, div = IMG_MEAN.view(3, 1, 1).cuda(), IMG_STD.view(3, 1, 1).cuda()

    sums = [0.0] * 4
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            inp = batch['inp'].cuda(non_blocking=True)
            gt = batch['gt'].cuda(non_blocking=True)
            pred = torch.sigmoid(model.infer(
                inp, batch['v'].cuda(non_blocking=True),
                batch['t_pos'].cuda(non_blocking=True), batch['t_neg'].cuda(non_blocking=True),
            ))
            sm, em, wfm, mae = utils.calc_cod(pred.float(), gt.float())
            B = inp.shape[0]
            sums[0] += sm * B
            sums[1] += em * B
            sums[2] += wfm * B
            sums[3] += mae * B
            cnt += B

            if args.save_maps:
                for i in range(B):
                    name = batch['id'][i]
                    th, tw = int(batch['shape'][i][0]), int(batch['shape'][i][1])
                    img_vis = (inp[i] * div + sub).clamp(0, 1)
                    pred_vis = pred[i:i + 1]  # soft [1,1008,1008]
                    gt_vis = gt[i:i + 1]
                    if (th, tw) != (img_vis.shape[1], img_vis.shape[2]):
                        img_vis = F.interpolate(img_vis.unsqueeze(0), size=(th, tw),
                                                 mode='bilinear', align_corners=False).squeeze(0)
                        pred_vis = F.interpolate(pred_vis, size=(th, tw),
                                                 mode='bilinear', align_corners=False)
                        gt_vis = F.interpolate(gt_vis, size=(th, tw), mode='nearest')
                    if args.postproc:
                        # 原始管线后处理（原分辨率）：σ2 高斯 -> 0.5 -> 开2 -> 闭3 -> 去碎<64
                        soft = np.ascontiguousarray(pred_vis[0, 0].float().cpu().numpy())  # [th,tw]
                        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=2)
                        bin_ = (soft > 0.5).astype(np.uint8)
                        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k2, iterations=2)
                        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k2, iterations=3)
                        n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, 8)
                        for j in range(1, n):
                            if stats[j, cv2.CC_STAT_AREA] < 64:
                                bin_[lab == j] = 0
                        pred_vis = torch.from_numpy(bin_).float().to(pred.device)[None]  # [1,th,tw]
                    else:
                        pred_vis = (pred_vis > 0.5).float()  # 二值图，与 gt 同格式
                    save_image(img_vis, os.path.join(out_dir, 'image', f'{name}.png'))
                    save_image(pred_vis, os.path.join(out_dir, 'pred', f'{name}.png'))
                    save_image(gt_vis, os.path.join(out_dir, 'gt', f'{name}.png'))
                    if args.overlay:
                        # 浅红叠加（早期管线同款：255,102,102, α=0.45）
                        red = torch.tensor([255, 102, 102], device=img_vis.device).float() / 255
                        m = pred_vis[0]  # [H,W] 二值
                        ov = img_vis * (1 - 0.45 * m[None]) + red[:, None, None] * (0.45 * m[None])
                        save_image(ov.clamp(0, 1), os.path.join(out_dir, 'overlay', f'{name}.png'))

    sm, em, wfm, mae = [s / cnt for s in sums]
    print(f"=== {ck_path} ===")
    print(f"Sm={sm:.4f}  Em={em:.4f}  FwBeta={wfm:.4f}  MAE={mae:.4f}  ({cnt} images)")
    if args.save_maps:
        print(f"image/gt/pred maps saved to {out_dir}/{{image,gt,pred}}")


if __name__ == '__main__':
    main()
