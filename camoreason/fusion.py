"""Region-level adaptive multimodal fusion (paper Eq.16-23, as published).

The fixed gamma of vanilla fusion
    Output = (1 - gamma) * Image_Feature + gamma * Text_Feature
assumes one scalar weight for the whole image; COD is a local-uncertainty
task, so the paper upgrades gamma to per-level, per-position, per-channel
reliability weights over the modalities M = {v, f, hp, t} (use_hp=True
adds the paper Eq.15 high-pass response A_hp as a fusion modality; with
use_hp=False the layout is the original M = {v, f, t}):

    Uv = P_s ⊙ sigma(W_gamma,s z+) + tanh(W_beta,s z+)        (Eq.16)
    Uf = Pi_f,s(Ff)                                          (Eq.17)
    Uhp = Pi_hp,s(A_hp)                                      (Eq.15, use_hp=True)
    Ut = Pi_t,s(z+) broadcast                                (Eq.18)
    Gsp = g_sp([Uv, Uf, (Uhp,) Ut])                          (Eq.19)
    Gch = g_ch([Pool(Uv), Pool(Uf), (Pool(Uhp),) Pool(Ut)])  (Eq.20)
    alpha_m = Softmax_m(Gsp + Gch)                           (Eq.21)
    F_fuse = sum_m alpha_m ⊙ U_m                             (Eq.22)

with the gate regularizer of Eq.23 (distribution + TV) applied by the
caller (losses.gate_reg).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MODAL = 3  # legacy M = {v, f, t} layout; use_hp=True runs 4 modalities


class RegionAdaptiveFusion(nn.Module):
    def __init__(self, ch=256, d_multi=512, ff_ch=128, n_levels=3, gate_hidden=64,
                 mode='gate', use_freq=True, use_text=True, use_hp=False):
        """mode: 'gate' (Eq.16-23 adaptive fusion) or 'fixed' (ablation:
        constant 1/M weights, the gate's step-0 init, no alphas returned).
        use_freq: False -> frequency modality dropped.
        use_text: False -> text modality dropped.
        use_hp: True -> A_hp high-pass response added as modality 4 (paper
        Eq.15); the projection is zero-init like every other modality.
        Modalities are M = {Uv} (+Uf if use_freq) (+Uhp if use_hp) (+Ut if
        use_text); with only Uv (paper Table 3 single-modal) the fusion
        degenerates to the visual branch Uv alone: no gate, no alphas."""
        super().__init__()
        self.mode = mode
        self.use_freq = use_freq
        self.use_text = use_text
        self.use_hp = use_hp
        self.n_mod = 1 + int(use_freq) + int(use_hp) + int(use_text)
        # Uv (Eq.16): text-FiLM over the neck level
        self.films = nn.ModuleList([nn.Linear(d_multi, 2 * ch) for _ in range(n_levels)])
        # Uf (Eq.17): frequency feature -> ch per level
        self.proj_fs = (
            nn.ModuleList([nn.Conv2d(ff_ch, ch, 1) for _ in range(n_levels)])
            if use_freq else None
        )
        # Uhp (Eq.15): high-pass response -> ch per level
        self.proj_hps = (
            nn.ModuleList([nn.Conv2d(1, ch, 1) for _ in range(n_levels)])
            if use_hp else None
        )
        # Ut (Eq.18): semantic embedding -> ch per level (broadcast)
        self.proj_ts = (
            nn.ModuleList([nn.Linear(d_multi, ch) for _ in range(n_levels)])
            if use_text else None
        )
        if mode == 'gate' and self.n_mod > 1:
            # Gsp (Eq.19): spatial reliability logits over the modalities
            self.spatial_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.n_mod * ch, gate_hidden, 1),
                        nn.GELU(),
                        nn.Conv2d(gate_hidden, self.n_mod, 1),
                    )
                    for _ in range(n_levels)
                ]
            )
            # Gch (Eq.20): channel reliability logits from pooled modalities
            self.channel_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.n_mod * ch, gate_hidden),
                        nn.GELU(),
                        nn.Linear(gate_hidden, self.n_mod),
                    )
                    for _ in range(n_levels)
                ]
            )
            with torch.no_grad():
                for gates in (self.spatial_gates, self.channel_gates):
                    for g in gates:
                        g[-1].weight.zero_()
                        if self.use_hp:
                            # 4-modality: visual-prior bias so alpha_v = 1 at
                            # step 0 -> F_fuse = Uv = V (identity init; the
                            # added modality earns its weight from gradients).
                            g[-1].bias.copy_(torch.tensor([4.0, -4.0, -4.0, -4.0]))
                        else:
                            # 3-modality protocol: uniform 1/M
                            g[-1].bias.zero_()
        else:
            self.spatial_gates = None
            self.channel_gates = None
        with torch.no_grad():
            for film in self.films:
                film.weight.zero_()
                film.bias.zero_()
                film.bias[:ch].fill_(6.0)  # sigmoid(6) ~= 1 -> Uv = V at step 0
            # zero-init the modality projections too: at step 0 Uf = Uhp = Ut = 0,
            # so F_fuse depends only on the visual branch (the adapter's own
            # zero/identity-init principle; random init here used to inject
            # untrained noise into the decoder input at gate weight)
            for proj in (self.proj_fs, self.proj_hps, self.proj_ts):
                if proj is not None:
                    for m in proj:
                        m.weight.zero_()
                        m.bias.zero_()

    def forward(self, V_list, Ff, z_pos, A_hp=None):
        """V_list: neck levels [B,256,H_s,W_s]; Ff: [B,128,H/4,W/4] fp32 or None
        (use_freq=False); z_pos: [B,d_multi]; A_hp: [B,1,H,W] fp32 or None
        (use_hp=False).
        Returns (F_fuse_list, alpha_list ([B,M,H,W] each) or None in fixed mode
        / single-modal 1-modality mode)."""
        fused, alphas = [], []
        for s, V in enumerate(V_list):
            B, C, H, W = V.shape
            g, b = self.films[s](z_pos.to(V.dtype)).chunk(2, dim=-1)
            Uv = V * torch.sigmoid(g)[:, :, None, None] + torch.tanh(b)[:, :, None, None]  # Eq.16
            mods = [Uv]
            if self.use_freq:
                Uf = self.proj_fs[s](
                    F.interpolate(Ff.to(V.dtype), size=(H, W), mode="bilinear", align_corners=False)
                )  # Eq.17
                mods.append(Uf)
            if self.use_hp:
                Uhp = self.proj_hps[s](
                    F.interpolate(A_hp.to(V.dtype), size=(H, W), mode="bilinear", align_corners=False)
                )  # Eq.15 high-pass response modality
                mods.append(Uhp)
            if self.use_text:
                Ut = self.proj_ts[s](z_pos.to(V.dtype))[:, :, None, None].expand(B, C, H, W)  # Eq.18
                mods.append(Ut)
            if self.n_mod == 1:
                # single-modal (paper Table 3): the visual branch alone, no gate
                fused.append(Uv)
                continue
            cat = torch.cat(mods, dim=1)  # [B, M*C, H, W]

            if self.mode == 'fixed':
                # w/o Adaptive Gate ablation: constant 1/M weights (= the gate's
                # step-0 initialization), no alpha, no gate loss downstream.
                F_fuse = cat[:, :C]
                for m in range(1, self.n_mod):
                    F_fuse = F_fuse + cat[:, m * C:(m + 1) * C]
                F_fuse = F_fuse / self.n_mod
                fused.append(F_fuse)
                continue

            Gsp = self.spatial_gates[s](cat)  # [B,M,H,W] Eq.19
            Gch = self.channel_gates[s](cat.mean(dim=(-2, -1)))[:, :, None, None]  # Eq.20
            alpha = torch.softmax(Gsp + Gch, dim=1)  # [B,M,H,W] Eq.21
            F_fuse = sum(
                alpha[:, m:m + 1] * cat[:, m * C:(m + 1) * C] for m in range(self.n_mod)
            )  # Eq.22
            fused.append(F_fuse)
            alphas.append(alpha)
        return fused, alphas if (self.mode == 'gate' and self.n_mod > 1) else None
