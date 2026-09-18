"""MultimodalAdapter: the complete CamoReason adaptation package on the frozen
SAM3 encoder — semantic projection, in-trunk injection, region-modal fusion
ALL live in this one module (the only trainable module group attached to
the frozen backbone; CamoReason just calls it).

The ViT offers three hook sites (models/sam3/model/vitdet.py):
  prepare(inp)                 once per forward, before the block loop
  pre_block(x, i)              before each block   (token injection + adapter
                                branch A_l(X_{l-1}), returns (x, a))
  post_block(h, a, i)          after each block    (paper Eq.14: h + σ(γ)⊙a
                                + tanh(β))

Outside the trunk, CamoReason.forward calls:
  project_semantic(v, t_pos, t_neg) -> (z+, z-)              paper Eq.5-6
  fuse(V_list, z_pos) -> (F_fuse, alphas)                    paper Eq.16-23

Everything trainable here is zero/identity-initialized, so at step 0 the
trunk computes exactly its native SAM3 features; adaptation is learned from
zero on top of the fully pretrained encoder (LoRA/Houlsby-style).

prepare() runs under autocast-disabled fp32: torch.fft has no bf16 support
and the frequency responses are numerically delicate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from camoreason.frequency import FrequencyPrompt
from camoreason.fusion import RegionAdaptiveFusion


class SemanticProjector(nn.Module):
    """CLIP embeddings -> multimodal semantic embedding.
    v = Norm(E_I(I)), t+ = Norm(E_T(p+)), t- = Norm(E_T(p-)) are cached (frozen CLIP).
    z+ = Norm(W_z [v || t+ || v * t+])   conditions everything downstream
    z- = Norm(W_n t-)                     used by L_con and the negative modality Un
    """

    def __init__(self, clip_dim=768, d_multi=512):
        super().__init__()
        self.W_z = nn.Linear(3 * clip_dim, d_multi)
        self.W_n = nn.Linear(clip_dim, d_multi)

    def forward(self, v, t_pos, t_neg):
        z_pos = F.normalize(
            self.W_z(torch.cat([v.float(), t_pos.float(), v.float() * t_pos.float()], dim=-1)),
            dim=-1,
            eps=1e-6,  # zero vector -> zero vector (no NaN) instead of 0/0
        )
        z_neg = F.normalize(self.W_n(t_neg.float()), dim=-1, eps=1e-6)
        return z_pos, z_neg


class BottleneckAdapter(nn.Module):
    """Houlsby-style bottleneck: A(X) = W_up δ(W_down LN(X)), up zero-init (paper Eq.12)."""

    def __init__(self, dim=1024, bottleneck=32):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(self.ln(x))))


def _bc(vec, x):
    """Broadcast a [B, C] per-sample vector onto x: [B, ..., C]."""
    return vec.view(vec.shape[0], *([1] * (x.dim() - 2)), vec.shape[1])


class FiLMHead(nn.Module):
    """z+ -> (gamma, beta) raw modulation params (paper Eq.13).

    Last layer zero-init: gamma = beta = 0 at step 0, and the bottleneck
    adapter output is also zero at step 0, so the block stays exact-identity.
    """

    def __init__(self, cond_dim=512, out_dim=1024, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * out_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, x):
        """Returns (gamma, beta) [B, out_dim] in x's dtype."""
        gamma, beta = self.mlp(z.to(x.dtype)).chunk(2, dim=-1)
        return gamma, beta


def _token_mlp(in_dim, hidden, out_dim):
    """Token head whose last layer is zero-init (token is exactly 0 at step 0)."""
    mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class MultimodalAdapter(nn.Module):
    """All CamoReason adaptation logic on the frozen SAM3 ViT trunk, in one module.

    In-trunk hooks (called by the ViT):
      - set_conditioning(z_pos): store the CLIP-derived semantic embedding.
      - prepare(image): frequency decomposition; text-conditioned band selection
        alpha_f = Softmax(g_f(z+)); weighted maps -> Ff (dense, 128ch @ H/4) and
        A_hp (high-pass, 1ch @ H); group-pooled boundary / texture descriptors.
      - pre_block(x, i): at the first block of each stage, add the stage token
        psi_s(z+), the boundary token and the texture token.
      - post_block(x, i): FiLM(z+) modulation + BottleneckAdapter residual.

    Outside the trunk (called by CamoReason.forward):
      - project_semantic(v, t_pos, t_neg) -> (z+, z-)              Eq.5-6
      - fuse(V_list, z_pos) -> (F_fuse, alphas)                    Eq.16-23
    """

    def __init__(self, img_size=1008, embed_dim=1024, d_multi=512, depth=32,
                 n_stages=4, bottleneck=32, film_hidden=256, ff_ch=128,
                 token_hidden=256, n_fpn_levels=3, clip_dim=768):
        super().__init__()
        assert depth % n_stages == 0
        self.depth = depth
        self.n_stages = n_stages
        self.depth_per_stage = depth // n_stages
        self.stage_first = set(s * self.depth_per_stage for s in range(n_stages))

        # frequency branch (fp32; conv weights fp32, run under disabled autocast)
        self.freq = FrequencyPrompt(img_size=img_size, out_ch=ff_ch)
        # text-controlled band selector g_f: z+ -> 16 band logits
        self.band_selector = nn.Sequential(
            nn.Linear(d_multi, 64), nn.GELU(), nn.Linear(64, FrequencyPrompt.N_MAPS)
        )
        # group descriptors: boundary (HIGH group) / texture (MID group) tokens
        n_high = len(FrequencyPrompt.HIGH_IDX)
        n_mid = len(FrequencyPrompt.MID_IDX)
        self.stage_tokens = nn.ModuleList(
            [_token_mlp(d_multi, token_hidden, embed_dim) for _ in range(n_stages)]
        )
        self.boundary_token = _token_mlp(n_high, token_hidden, embed_dim)
        self.texture_token = _token_mlp(n_mid, token_hidden, embed_dim)

        # per-block modules
        self.bottlenecks = nn.ModuleList(
            [BottleneckAdapter(embed_dim, bottleneck) for _ in range(depth)]
        )
        self.films = nn.ModuleList(
            [FiLMHead(d_multi, embed_dim, film_hidden) for _ in range(depth)]
        )

        # semantic projection (Eq.5-6) and region-modal fusion (Eq.16-23)
        self.semantic = SemanticProjector(clip_dim=clip_dim, d_multi=d_multi)
        self.fusion = RegionAdaptiveFusion(d_multi=d_multi, ff_ch=ff_ch, n_levels=n_fpn_levels)

        # runtime state (filled by set_conditioning / prepare each forward)
        self.z_pos = None
        self.Ff = None        # [B, ff_ch, H/4, W/4] fp32
        self.A_hp = None      # [B, 1, H, W] fp32
        self._tok_b = None
        self._tok_t = None

    def set_conditioning(self, z_pos):
        self.z_pos = z_pos

    def prepare(self, image):
        """image: [B,3,H,W] (network input, any dtype). Runs fp32."""
        with torch.autocast(device_type=image.device.type, enabled=False):
            z = self.z_pos.float() if self.z_pos is not None else None
            assert z is not None, "set_conditioning(z_pos) must be called before forward"
            gray = image.float().mean(dim=1, keepdim=True)  # [B,1,H,W]
            maps, hp = self.freq.decompose(gray)           # [B,16,H,W], [B,1,H,W]

            # text-controlled band selection (Eq. spec 2.4: e.g. "striped snake"
            # attends to stripe continuity, "stonefish" to contour breaks)
            alpha_f = torch.softmax(self.band_selector(z), dim=-1)  # [B,16]
            weighted = maps * alpha_f[:, :, None, None]              # [B,16,H,W]

            self.Ff = self.freq.project(weighted)                    # [B,128,H/4,W/4]
            self.A_hp = hp

            # group descriptors (global pooled band energies -> tokens)
            b_pool = weighted[:, list(self.freq.HIGH_IDX)].mean(dim=(-2, -1))  # [B,11]
            t_pool = weighted[:, list(self.freq.MID_IDX)].mean(dim=(-2, -1))    # [B,4]
            self._tok_b = self.boundary_token(b_pool)  # [B, embed_dim]
            self._tok_t = self.texture_token(t_pool)  # [B, embed_dim]

    def pre_block(self, x, i):
        """Token injection at the first block of each stage (i in {0,8,16,24})
        plus the paper Eq.12 adapter branch on the block input X_{l-1}.
        Returns (x, a) with a = A_l(x)."""
        if i in self.stage_first:
            s = i // self.depth_per_stage
            tok = (
                self.stage_tokens[s](self.z_pos.to(x.dtype))
                + self._tok_b.to(x.dtype)
                + self._tok_t.to(x.dtype)
            )  # [B, C]; zero at step 0 (all three heads zero-init)
            x = x + _bc(tok, x)
        a = self.bottlenecks[i](x)
        return x, a

    def post_block(self, h, a, i):
        """Paper Eq.14: X_l = T_l(X_{l-1}) + σ(γ)⊙A_l(X_{l-1}) + tanh(β).
        h is the frozen block output, a is the adapter branch from pre_block."""
        gamma, beta = self.films[i](self.z_pos, h)
        return h + _bc(torch.sigmoid(gamma), h) * a + _bc(torch.tanh(beta), h)

    # ------------------------------------------------------------------ #
    # outside-the-trunk API (called by CamoReason.forward)
    # ------------------------------------------------------------------ #
    def project_semantic(self, v, t_pos, t_neg):
        """CLIP cache embeddings -> (z+, z-) semantic embedding (Eq.5-6)."""
        return self.semantic(v, t_pos, t_neg)

    def fuse(self, V_list, z_pos):
        """Region-level adaptive fusion over the encoder output (Eq.16-23).
        Requires prepare() to have run during the ViT forward: Ff is filled
        there. Returns (F_fuse_list, alpha_list ([B,3,H,W] each))."""
        return self.fusion(V_list, self.Ff, z_pos)
