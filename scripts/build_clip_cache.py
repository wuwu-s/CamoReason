"""Build the plain-caption CLIP cache for the w/o CamoReasoner ablation.

Paper Table 2 "w/o CamoReasoner": every sample gets the SAME fixed generic
caption pair instead of the cached Qwen camouflage-reasoner output:
    p+ = "a camouflaged object"   p- = "background"
v (the CLIP image embedding) is still per-sample and identical to the full
cache. Rows are ordered train.txt, then val.txt, then test.txt (5132 rows),
so the cache can be swapped in via the dataset's clip_cache= argument.

The ORIGINAL reasoner cache (clip_cache.npz) was built with this script's
previous version (compose_p_pos/compose_p_neg from camoreason.clip_cache,
module since deleted); it already exists on disk and is NOT regenerated here.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PLAIN_POS = "a camouflaged object"
PLAIN_NEG = "background"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/MHCD2026")
    ap.add_argument("--clip", default="/root/autodl-tmp/clip")
    ap.add_argument("--out", default="/root/autodl-tmp/MHCD2026/clip_cache_noreasoner.npz")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.clip).cuda().eval()
    processor = CLIPProcessor.from_pretrained(args.clip)

    # fixed caption embeddings, computed once (identical for every sample)
    with torch.no_grad():
        txt_p_inp = processor(text=[PLAIN_POS], return_tensors="pt",
                              padding=True, truncation=True, max_length=77).to("cuda")
        txt_n_inp = processor(text=[PLAIN_NEG], return_tensors="pt",
                              padding=True, truncation=True, max_length=77).to("cuda")
        tp0 = F.normalize(model.get_text_features(**txt_p_inp), dim=-1).cpu().half()
        tn0 = F.normalize(model.get_text_features(**txt_n_inp), dim=-1).cpu().half()
    print(f"plain captions: p+={PLAIN_POS!r} p-={PLAIN_NEG!r}")

    all_v, all_tp, all_tn = [], [], []
    with torch.no_grad():
        for split in ("train", "val", "test"):
            with open(os.path.join(args.root, f"{split}.txt")) as f:
                ids = [line.strip() for line in f if line.strip()]
            for i in tqdm(range(0, len(ids), args.batch), desc=f"clip:{split}"):
                batch_ids = ids[i:i + args.batch]
                images = [
                    Image.open(os.path.join(args.root, split, "Image", f"{id_}.jpg")).convert("RGB")
                    for id_ in batch_ids
                ]
                img_inp = processor(images=images, return_tensors="pt").to("cuda")
                v = model.get_image_features(**img_inp)
                v = F.normalize(v, dim=-1)
                all_v.append(v.cpu().half())
                all_tp.append(tp0.expand(len(batch_ids), -1))
                all_tn.append(tn0.expand(len(batch_ids), -1))

    v = torch.cat(all_v)
    tp = torch.cat(all_tp)
    tn = torch.cat(all_tn)
    np.savez_compressed(args.out, v=v.numpy(), t_pos=tp.numpy(), t_neg=tn.numpy())
    print(f"saved {args.out}: v={tuple(v.shape)} t_pos={tuple(tp.shape)} t_neg={tuple(tn.shape)}")
    print(f"sanity: mean cos(v,t+)={(v * tp).sum(-1).mean():.4f}  "
          f"mean cos(v,t-)={(v * tn).sum(-1).mean():.4f}")


if __name__ == "__main__":
    main()
