# CamoReason on MHCD2026

Multimodal prompt-based camouflage segmentation: frozen SAM3 ViT-H encoder conditioned by
Qwen camouflage-reasoning text prompts (CLIP-encoded) and frequency prompts, with
region-adaptive multimodal fusion and a SAM3 mask decoder. The model lives in the
`camoreason/` package; everything runs via `python -m camoreason.*`.

## Model

| Module | Role |
|---|---|
| `camoreason/model.py` | CamoReason: frozen SAM3 encoder + adapter + MaskDecoder + losses |
| `camoreason/adapter.py` | MultimodalAdapter: semantic projection (z+, z-), in-trunk injection, fusion wiring |
| `camoreason/fusion.py` | RegionAdaptiveFusion: modalities {v, f, hp, t} (use_hp adds the Eq.15 high-pass response) |
| `camoreason/frequency.py` | FrequencyPrompt: FFT bands / DWT / LoG / Gabor (16 maps), high/low-pass modes |
| `camoreason/losses.py` | focal+dice, soft-IoU MSE, InfoNCE, gate regularization |

## Configs

| Config | Purpose |
|---|---|
| `configs/camoreason-mhcd.yaml` | Main protocol: tau 0.07, 50 epochs, bs8 x accum3, lr 4.5e-4 cosine |
| `configs/camoreason-mhcd-ahp.yaml` | tau 0.5 + use_hp 4-modality + dedicated positive head W_p |
| `configs/camoreason-mhcd-ahp-resume.yaml` | Resume variant (LR continues the original cosine curve exactly) |
| `configs/camoreason-mhcd-ahp-hp10.yaml` | 4-modality model under the 10-epoch sweep protocol |
| `configs/camoreason-mhcd-ab_no_{image,freq,gate,reasoner,text_freq}.yaml` | Table 2/3 ablations (20 epochs) |
| `configs/camoreason-mhcd-hp_*.yaml` | Table 4 hyper-parameter sweep (10 epochs) |

## Train / Eval

```bash
PY=/root/miniconda3/envs/sam3/bin/python   # adjust to your env
$PY -m camoreason.train --config configs/camoreason-mhcd.yaml --name _camoreason-mhcd
$PY -m camoreason.eval  --config configs/camoreason-mhcd.yaml --name _camoreason-mhcd \
    --save-maps --out-dir save/results_main
```

Checkpoints land in `save/<name>/` (`model_epoch_best.pth` / `model_epoch_last.pth`).

## Data

MHCD2026 (train 3284 / val 823 / test 1025): images + SAM3-zero-shot pseudo-GT masks +
Qwen prompt JSONs. CLIP embeddings are cached in `clip_cache.npz` (v / t_pos / t_neg, rows
ordered train, val, test).
