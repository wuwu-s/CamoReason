"""Generate the paper Table 4 hyper-parameter sweep configs (10-epoch protocol).

Table 4 grid (paper best in bold, that row = the main model's setting):
  multimodal dimension dmulti : 256 / **512** / 1024
  bottleneck dimension        : 8 / 16 / **32** / 64
  prompt type                 : **high-pass** / low-pass
  frequency ratio r_freq      : 0.125 / **0.25** / 0.375 / 0.5

9 runs to train; each keeps the main protocol (bs8/accum3/lr4.5e-4/warmup5/
epoch_val2) with ONLY the swept value changed, at epoch_max 10 (user decision:
sweep compares relative order, absolute values not used in the main table).

Emits configs/camoreason-mhcd-hp_<key>.yaml and the matching overfit smoke
configs/camoreason-overfit-hp_<key>.yaml (limit 50/8, 4ep, warmup 0, accum 1).
Regenerate anytime with:  python scripts/gen_sweep_configs.py

The sweep protocol is tau=0.07 with early stop disabled; the guard below
refuses to regenerate configs if the base config drifts from it.
"""
import copy
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

BASE = os.path.join(REPO_ROOT, "configs", "camoreason-mhcd.yaml")

# (key, model-args override dict)
VARIANTS = [
    ("dmulti256",  {"d_multi": 256}),
    ("dmulti1024", {"d_multi": 1024}),
    ("bn8",        {"bottleneck": 8}),
    ("bn16",       {"bottleneck": 16}),
    ("bn64",       {"bottleneck": 64}),
    ("prompt_low", {"prompt_type": "low"}),
    ("rfreq0125",  {"r_freq": 0.125}),
    ("rfreq0375",  {"r_freq": 0.375}),
    ("rfreq05",    {"r_freq": 0.5}),
]


def main():
    with open(BASE) as f:
        base = yaml.load(f, Loader=yaml.FullLoader)

    # protocol freeze guard (see docstring): the ran sweep = tau 0.07, no early_stop
    tau = base["model"]["args"].get("tau")
    es = base.get("early_stop", 0)
    assert tau == 0.07 and es == 0, (
        f"base config drifted from the Table 4 sweep protocol "
        f"(tau={tau}, early_stop={es}); expected tau=0.07, early_stop disabled. "
        f"Regenerating now would silently change the sweep protocol."
    )

    for key, over in VARIANTS:
        # ---- training config: 10-epoch protocol, one swept value changed ----
        cfg = copy.deepcopy(base)
        cfg["epoch_max"] = 10
        cfg["model"]["args"].update(over)
        tag = f"paper Table 4 sweep: {' '.join(f'{k}={v}' for k, v in over.items())}"
        cfg.setdefault("_comment", tag)
        out = os.path.join(REPO_ROOT, "configs", f"camoreason-mhcd-hp_{key}.yaml")
        with open(out, "w") as f:
            f.write(f"# {tag}\n")
            yaml.dump(cfg, f, sort_keys=False)
        print(f"wrote {os.path.relpath(out, REPO_ROOT)}")

        # ---- overfit smoke config (same variant, tiny data) ----
        sm = copy.deepcopy(cfg)
        for split in ("train_dataset", "val_dataset", "test_dataset"):
            sm[split]["num_workers"] = 2
            sm[split]["args"]["limit"] = 50 if split == "train_dataset" else 8
        sm["epoch_max"] = 4
        sm["epoch_val"] = 1
        sm["warmup_epochs"] = 0
        sm["accum_steps"] = 1
        out = os.path.join(REPO_ROOT, "configs", f"camoreason-overfit-hp_{key}.yaml")
        with open(out, "w") as f:
            f.write(f"# Overfit smoke for {tag} (diagnostic only, NOT paper protocol).\n")
            yaml.dump(sm, f, sort_keys=False)
        print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
