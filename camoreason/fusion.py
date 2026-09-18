"""Region-level adaptive multimodal fusion (paper Eq.16-23, as published).

The fixed gamma of vanilla fusion
    Output = (1 - gamma) * Image_Feature + gamma * Text_Feature
assumes one scalar weight for the whole image; COD is a local-uncertainty
task, so the paper upgrades gamma to per-level, per-position, per-channel
reliability weights over THREE modalities M = {v, f, t}:

    Uv = P_s ⊙ sigma(W_gamma,s z+) + tanh(W_beta,s z+)        (Eq.16)
    Uf = Pi_f,s(Ff)                                          (Eq.17)
    Ut = Pi_t,s(z+) broadcast                                (Eq.18)
    Gsp = g_sp([Uv, Uf, Ut])                                 (Eq.19)
    Gch = g_ch([Pool(Uv), Pool(Uf), Pool(Ut)])               (Eq.20)
    alpha_m = Softmax_m(Gsp + Gch)                           (Eq.21)
    F_fuse = sum_m alpha_m ⊙ U_m                             (Eq.22)

with the gate regularizer of Eq.23 (distribution + TV) applied by the
caller (losses.gate_reg).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MODAL = 3  # M = {v, f, t}: visual / frequency / semantic (Eq.18)


class RegionAdaptiveFusion(nn.Module):
    def __init__(self, ch=256, d_multi=512, ff_ch=128, n_levels=3, gate_hidden=64):
        super().__init__()
        # Uv (Eq.16): text-FiLM over the neck level
        self.films = nn.ModuleList([nn.Linear(d_multi, 2 * ch) for _ in range(n_levels)])
        # Uf (Eq.17): frequency feature -> ch per level
        self.proj_fs = nn.ModuleList([nn.Conv2d(ff_ch, ch, 1) for _ in range(n_levels)])
        # Ut (Eq.18): semantic embedding -> ch per level (broadcast)
        self.proj_ts = nn.ModuleList([nn.Linear(d_multi, ch) for _ in range(n_levels)])
        # Gsp (Eq.19): spatial reliability logits over the 3 modalities
        self.spatial_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(N_MODAL * ch, gate_hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(gate_hidden, N_MODAL, 1),
                )
                for _ in range(n_levels)
            ]
        )
        # Gch (Eq.20): channel reliability logits from pooled modalities
        self.channel_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(N_MODAL * ch, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, N_MODAL),
                )
                for _ in range(n_levels)
            ]
        )
        with torch.no_grad():
            for film in self.films:
                film.weight.zero_()
                film.bias.zero_()
                film.bias[:ch].fill_(6.0)  # sigmoid(6) ~= 1 -> Uv = V at step 0
            for gates in (self.spatial_gates, self.channel_gates):
                for g in gates:
                    g[-1].weight.zero_()
                    g[-1].bias.zero_()  # uniform alpha = 1/3 at step 0

    def forward(self, V_list, Ff, z_pos):
        """V_list: neck levels [B,256,H_s,W_s]; Ff: [B,128,H/4,W/4] fp32;
        z_pos: [B,d_multi]. Returns (F_fuse_list, alpha_list ([B,3,H,W] each))."""
        fused, alphas = [], []
        for s, V in enumerate(V_list):
            B, C, H, W = V.shape
            g, b = self.films[s](z_pos.to(V.dtype)).chunk(2, dim=-1)
            Uv = V * torch.sigmoid(g)[:, :, None, None] + torch.tanh(b)[:, :, None, None]  # Eq.16
            Uf = self.proj_fs[s](
                F.interpolate(Ff.to(V.dtype), size=(H, W), mode="bilinear", align_corners=False)
            )  # Eq.17
            Ut = self.proj_ts[s](z_pos.to(V.dtype))[:, :, None, None].expand(B, C, H, W)  # Eq.18
            cat3 = torch.cat([Uv, Uf, Ut], dim=1)  # [B, 3C, H, W]
            Gsp = self.spatial_gates[s](cat3)  # [B,3,H,W] Eq.19
            Gch = self.channel_gates[s](cat3.mean(dim=(-2, -1)))[:, :, None, None]  # Eq.20
            alpha = torch.softmax(Gsp + Gch, dim=1)  # [B,3,H,W] Eq.21
            F_fuse = sum(
                alpha[:, m : m + 1] * cat3[:, m * C : (m + 1) * C] for m in range(N_MODAL)
            )  # Eq.22
            fused.append(F_fuse)
            alphas.append(alpha)
        return fused, alphas
