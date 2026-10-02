"""CamoReason-SAM training (mirrors the original repo's train.py; paper protocol).

Structure copied from the repo-root train.py: make_data_loader(s) / eval_psnr /
prepare_training / train / save / main; --config/--name/--tag; utils.set_save_path
(log.txt + tensorboard); config.yaml dump; model_epoch_last.pth every epoch +
model_epoch_best.pth on best val Sm.

Deliberate deviations (all forced by hardware / paper / user choices):
  1. single GPU, no DDP (this machine is 1x32GB; the original hard-requires NCCL)
  2. LambdaLR stepped per optimizer step: 5-epoch linear warmup -> cosine to 1e-6
     (paper; the original steps CosineAnnealingLR per epoch with no warmup)
  3. gradient accumulation 3 (paper effective batch 24 = 8 x 3) + a final step on
     the trailing partial window (e.g. 445 batches % 3 == 1)
  4. bf16 autocast for training, fp32 no-autocast for validation (32GB cannot fit
     the fp32 forward; the paper used 48GB)
  5. weight loading = the original remap (detector.backbone.* -> image_encoder.*,
     tracker.sam_mask_decoder.* -> mask_decoder.*, pe_layer, no_mask_embed) with
     shape checks + strict=False; resume = point sam_checkpoint at a previous
     model_epoch_*.pth and set resume: N (fresh optimizer/scheduler, as original)
  6. freeze rule: startswith("image_encoder") and ".adapter." not in name;
     trainable = trunk adapter + semantic + fusion + mask_decoder
     + pe_layer + no_mask_embed
  7. epoch_val: 2 (user choice; original validates every epoch)
  8. forward convention kept: loss = model(inp, v, t_pos, t_neg, gt); components
     in model.loss_log

Loss (paper Eq.31): L_seg(focal+dice, oracle k*) + lambda_iou*MSE-IoU
+ lambda_con*InfoNCE + lambda_gate*L_gate.
"""
import argparse
import math
import os
import sys

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import utils  # noqa: E402  (repo utilities: log, Timer, calc_cod, ...)
from camoreason.mhcd_dataset import MHCDDataset  # noqa: E402
from camoreason.model import CamoReason  # noqa: E402

config, log, writer = None, None, None


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = MHCDDataset(**spec['args'])
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if hasattr(v, 'shape'):
            log(f'  {k}: shape={tuple(v.shape)}')
        else:
            log(f'  {k}: {v}')
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=spec.get('num_workers', 8),
        pin_memory=True,
        drop_last=(tag == 'train'),
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    """fp32, no autocast (matches the original repo's metric protocol)."""
    model.eval()
    if eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    else:
        raise ValueError(f'unknown eval_type: {eval_type}')

    pbar = tqdm(total=len(loader), leave=False, desc='val')
    val_metric = [0.0, 0.0, 0.0, 0.0]
    cnt = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch['inp'].cuda(non_blocking=True)
            gt = batch['gt'].cuda(non_blocking=True)
            pred = torch.sigmoid(model.infer(
                inp, batch['v'].cuda(non_blocking=True),
                batch['t_pos'].cuda(non_blocking=True), batch['t_neg'].cuda(non_blocking=True),
            ))
            results = metric_fn(pred.float(), gt.float())
            for i in range(4):
                val_metric[i] += results[i] * pred.shape[0]
            cnt += pred.shape[0]
            pbar.update(1)
    pbar.close()
    model.train()
    return (*[m / cnt for m in val_metric], metric1, metric2, metric3, metric4)


def load_sam3_weights(model):
    """The original repo's remap, applied verbatim (see docstring deviation 6)."""
    ckpt = torch.load(config['sam_checkpoint'], map_location='cpu', weights_only=False)
    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']

    new_state_dict = {}
    ref_state_dict = model.state_dict()
    print("Loading checkpoint with 'detector.backbone' / 'tracker.sam_*' remap ...")
    for k, v in ckpt.items():
        if k.startswith('detector.backbone.'):
            new_k = k.replace('detector.backbone.', 'image_encoder.')
        elif 'mask_decoder' in k:
            new_k = f"mask_decoder.{k.split('mask_decoder.')[-1]}"
        elif 'pe_layer' in k:
            new_k = f"pe_layer.{k.split('pe_layer.')[-1]}"
        elif 'no_mask_embed' in k:
            new_k = 'no_mask_embed.weight'
        else:
            new_k = k

        if new_k in ref_state_dict:
            if v.shape != ref_state_dict[new_k].shape:
                print(f"Warning: Skipping {new_k} due to shape mismatch. "
                      f"Ckpt: {v.shape} vs Model: {ref_state_dict[new_k].shape}")
                continue
            new_state_dict[new_k] = v
        elif new_k == k:
            pass  # e.g. tracker.* / detector.* keys unused by CamoReason

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"\nLoad result: {len(msg.missing_keys)} missing keys "
          f"({sum(1 for k in msg.missing_keys if '.adapter.' in k)} = new adapter modules, as expected).")
    if msg.missing_keys:
        print("Sample missing keys:", msg.missing_keys[:5])
    return model


def make_lr_lambda(cfg, steps_per_epoch):
    total = steps_per_epoch * cfg['epoch_max']
    warm = steps_per_epoch * cfg.get('warmup_epochs', 5)
    lr = cfg['optimizer']['args']['lr']
    eta_min = cfg.get('lr_min', 1e-6)

    def lmbd(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        target = eta_min + (lr - eta_min) * 0.5 * (1 + math.cos(math.pi * prog))
        return target / lr

    return lmbd


def prepare_training():
    model = CamoReason(**config['model']['args']).cuda()
    model = load_sam3_weights(model)

    # freeze the SAM3 main body; keep trunk.adapter + everything outside image_encoder
    for name, para in model.named_parameters():
        if name.startswith('image_encoder') and '.adapter.' not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params),
          '\nmodel_total_params:' + str(model_total_params))
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = utils.make_optimizer(train_params, config['optimizer'])
    epoch_start = 1
    if config.get('resume') is not None:
        epoch_start = config['resume'] + 1
    return model, optimizer, epoch_start, train_params


def train(train_loader, model, epoch):
    """One epoch under bf16 autocast with gradient accumulation (deviations 3-4)."""
    model.train()
    accum_steps = config.get('accum_steps', 3)
    optimizer = model.optimizer
    scheduler = model.scheduler

    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    accum = {}
    optimizer.zero_grad(set_to_none=True)

    for bi, batch in enumerate(train_loader):
        inp = batch['inp'].cuda(non_blocking=True)
        gt = batch['gt'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(inp, batch['v'].cuda(non_blocking=True),
                         batch['t_pos'].cuda(non_blocking=True),
                         batch['t_neg'].cuda(non_blocking=True), gt)
        # NaN guard: fail loudly at the first non-finite loss instead of
        # silently poisoning all weights.
        if not torch.isfinite(loss):
            dump_path = os.path.join(REPO_ROOT, 'save',
                                     f'nan_dump_ep{epoch}_it{bi}.pth')
            print(f'[NAN GUARD] ep{epoch} it{bi} ids={list(batch["id"])} '
                  f'loss_log={ {k: float(v) for k, v in model.loss_log.items()} }',
                  flush=True)
            torch.save({
                'epoch': epoch, 'bi': bi, 'ids': list(batch['id']),
                'loss_log': {k: float(v) for k, v in model.loss_log.items()},
                'batch': {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                          for k, v in batch.items()},
                'state_dict': {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
            }, dump_path)
            raise RuntimeError(
                f'NaN loss ep{epoch} it{bi} ids={list(batch["id"])} '
                f'loss_log={ {k: float(v) for k, v in model.loss_log.items()} } '
                f'(batch+state saved to {dump_path})'
            )
        (loss / accum_steps).backward()
        for k, v in model.loss_log.items():
            accum[k] = accum.get(k, 0.0) + v.item() / accum_steps
        if (bi + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        pbar.update(1)
        pbar.set_postfix(seg=f"{accum.get('seg', 0.0) / (bi + 1):.3f}",
                         total=f"{accum.get('total', 0.0) / (bi + 1):.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # trailing partial accumulation window (e.g. 445 batches % 3 == 1): step once
    # more, otherwise the last window's gradients are dropped by next zero_grad
    if len(train_loader) % accum_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    pbar.close()
    n = max(1, len(train_loader))
    return {k: v / n for k, v in accum.items()}


def save(config, model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, train_params = prepare_training()
    model.optimizer = optimizer
    model.train_params = train_params
    steps_per_epoch = math.ceil(len(train_loader) / config.get('accum_steps', 3))
    # Offset the LR schedule so a resumed segment continues the original
    # cosine curve at the exact position: last_epoch = steps already taken,
    # so the first resumed step evaluates lmbd((epoch_start-1)*S + 1).
    resume_offset = (epoch_start - 1) * steps_per_epoch
    model.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(config, steps_per_epoch)
    )
    if resume_offset > 0:
        model.scheduler.last_epoch = resume_offset

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 2)
    early_stop = config.get('early_stop', 0)  # patience in epochs; 0 = disabled
    # carry over the pre-resume best val so a resumed run never overwrites
    # model_epoch_best.pth with a worse first val.
    max_val_v = float(config.get('resume_best_val', -1e18))
    best_epoch = epoch_start
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        losses = train(train_loader, model, epoch)
        save(config, model, save_path, 'last')

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('loss', {'train G': losses['total']}, epoch)

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(
                val_loader, model, eval_type=config.get('eval_type')
            )
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            log_info.append(
                'train G: loss={:.4f} (seg {:.3f} iou {:.3f} con {:.3f} gate {:.3f})'.format(
                    losses['total'], losses['seg'], losses['iou'], losses['con'],
                    losses['gate'],
                )
            )
            log_info.append('val: {}={:.4f}'.format(metric1, result1))
            writer.add_scalars(metric1, {'val': result1}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result2))
            writer.add_scalars(metric2, {'val': result2}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result3))
            writer.add_scalars(metric3, {'val': result3}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result4))
            writer.add_scalars(metric4, {'val': result4}, epoch)

            if result1 > max_val_v:  # best by val Sm (COD metric1)
                max_val_v = result1
                best_epoch = epoch
                save(config, model, save_path, 'best')

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()

            if early_stop > 0 and (epoch - best_epoch) >= early_stop:
                log(f'early stop: no val Sm improvement for {early_stop} epochs '
                    f'(best {max_val_v:.4f} at epoch {best_epoch})')
                break

    log('training done')
    log(f"best val sm={max_val_v:.4f}; ckpts in {save_path} (model_epoch_best/last.pth)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/camoreason-mhcd.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(REPO_ROOT, 'save', save_name)

    main(config, save_path, args=args)
