"""MHCD2026 dataset with merged CLIP cache loading.

Physical structure (paper 5.1: train/val 8:2 carved from the training pool):
  - train/  train.txt  3284 ids (train only; 280 empty-negative-prompt removed 2026-09-16)
  - val/    val.txt    823 ids (held out for model selection; 67 removed)
  - test/   test.txt   1025 ids (final evaluation only; 89 removed)
  Removed samples are quarantined in /root/autodl-tmp/MHCD2026_removed_negprompt/.

Each sample: {split}/Image/{id}.jpg + {split}/gt/{id}_mask.png +
{split}/prompt/{id}.json (the cached Qwen camouflage-reasoner output;
p+ = positive_prompt (fallback target_hypothesis),
p-  = negative_prompt + background_distractors).

CLIP embeddings (v / t_pos / t_neg, [N,768] fp16) are pre-cached in
clip_cache.npz with rows ordered train.txt, then val.txt, then test.txt.
The cache is loaded once here, so __getitem__ returns image/gt/v/t_pos/t_neg
directly and the training loop needs no separate lookup step. The total row
count is asserted to equal the sum of the three split files (5132) so a stale
cache cannot silently misalign with the data.

Image normalization is SAM3's own (x - 0.5) / 0.5: the ViT trunk is frozen and
was pretrained with this protocol, so the identity claim "step 0 = native
SAM3" holds under its native preprocessing (the SAM-Adapter repo's ImageNet
stats do not apply to a frozen SAM3 encoder).
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMG_MEAN = [0.5, 0.5, 0.5]  # SAM3 preprocessing (frozen trunk protocol)
IMG_STD = [0.5, 0.5, 0.5]

SPLITS = ("train", "val", "test")


def _read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class MHCDDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp/MHCD2026", split="train",
                 inp_size=1008, augment=False, limit=None,
                 clip_cache=None):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        self.root = root
        self.split = split
        self.inp_size = inp_size

        # CLIP-cache row offset: rows are ordered train.txt | val.txt | test.txt
        counts = {s: len(_read_ids(os.path.join(root, f"{s}.txt"))) for s in SPLITS}
        off = sum(counts[s] for s in SPLITS[: SPLITS.index(split)])
        ids = _read_ids(os.path.join(root, f"{split}.txt"))
        self.entries = [(id_, off + i) for i, id_ in enumerate(ids)]
        if limit:
            self.entries = self.entries[:limit]
        self.ids = [id_ for id_, _ in self.entries]

        # merged CLIP cache (v / t_pos / t_neg)
        cache_path = clip_cache or os.path.join(root, "clip_cache.npz")
        data = np.load(cache_path)
        self.clip_v = torch.from_numpy(data["v"].astype(np.float32))          # [N,768]
        self.clip_t_pos = torch.from_numpy(data["t_pos"].astype(np.float32))  # [N,768]
        self.clip_t_neg = torch.from_numpy(data["t_neg"].astype(np.float32))  # [N,768]
        total = sum(counts.values())
        assert self.clip_v.shape[0] == total, (
            f"CLIP cache has {self.clip_v.shape[0]} rows but splits sum to {total}; "
            "rebuild the cache with scripts/build_clip_cache.py"
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.augment = augment
        if augment:
            self.flip = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        id_, row = self.entries[idx]
        img = Image.open(os.path.join(self.root, self.split, "Image", f"{id_}.jpg")).convert("RGB")
        gt = Image.open(os.path.join(self.root, self.split, "gt", f"{id_}_mask.png")).convert("L")
        w, h = img.size  # PIL: (width, height), read before the resize transform
        if self.augment:
            img = self.flip(img)
            gt = self.flip(gt)
        return {
            "inp": self.img_transform(img),
            "gt": self.mask_transform(gt),  # [1,H,W] in {0,1} (NEAREST + ToTensor)
            "v": self.clip_v[row],          # [768]
            "t_pos": self.clip_t_pos[row],  # [768]
            "t_neg": self.clip_t_neg[row],  # [768]
            "id": id_,
            "shape": torch.tensor([h, w]),  # original resolution (eval.py protocol)
        }
