"""Precompute frozen CLIP embeddings for all images: v (image), t_pos (p+), t_neg (p-).

Paper Eq.5: v = Norm(E_I(I)), t+ = Norm(E_T(p+)), t- = Norm(E_T(p-)).
Rows are ordered train.txt (3565), then val.txt (891), then test.txt (1114);
saved as fp16 npz (v/t_pos/t_neg [N,768]).
The frozen encoder is run once here — training only reads the cache (zero CLIP cost).
"""
import argparse
import json
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

from camoreason.clip_cache import compose_p_neg, compose_p_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/MHCD2026")
    ap.add_argument("--clip", default="/root/autodl-tmp/clip")
    ap.add_argument("--out", default="/root/autodl-tmp/MHCD2026/clip_cache.npz")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.clip).cuda().eval()
    processor = CLIPProcessor.from_pretrained(args.clip)

    all_v, all_tp, all_tn = [], [], []
    with torch.no_grad():
        for split in ("train", "val", "test"):
            with open(os.path.join(args.root, f"{split}.txt")) as f:
                ids = [line.strip() for line in f if line.strip()]
            for i in tqdm(range(0, len(ids), args.batch), desc=f"clip:{split}"):
                batch_ids = ids[i:i + args.batch]
                images, texts_p, texts_n = [], [], []
                for id_ in batch_ids:
                    images.append(
                        Image.open(os.path.join(args.root, split, "Image", f"{id_}.jpg")).convert("RGB")
                    )
                    with open(os.path.join(args.root, split, "prompt", f"{id_}.json")) as f:
                        p = json.load(f)
                    texts_p.append(compose_p_pos(p))
                    texts_n.append(compose_p_neg(p))
                img_inp = processor(images=images, return_tensors="pt").to("cuda")
                txt_p_inp = processor(
                    text=texts_p, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to("cuda")
                txt_n_inp = processor(
                    text=texts_n, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to("cuda")
                v = model.get_image_features(**img_inp)
                tp = model.get_text_features(**txt_p_inp)
                tn = model.get_text_features(**txt_n_inp)
                v = F.normalize(v, dim=-1)
                tp = F.normalize(tp, dim=-1)
                tn = F.normalize(tn, dim=-1)
                all_v.append(v.cpu().half())
                all_tp.append(tp.cpu().half())
                all_tn.append(tn.cpu().half())

    v = torch.cat(all_v)
    tp = torch.cat(all_tp)
    tn = torch.cat(all_tn)
    np.savez_compressed(args.out, v=v.numpy(), t_pos=tp.numpy(), t_neg=tn.numpy())
    print(f"saved {args.out}: v={tuple(v.shape)} t_pos={tuple(tp.shape)} t_neg={tuple(tn.shape)}")
    print(f"sanity: mean cos(v,t+)={ (v * tp).sum(-1).mean():.4f}  "
          f"mean cos(v,t-)={(v * tn).sum(-1).mean():.4f}")


if __name__ == "__main__":
    main()
