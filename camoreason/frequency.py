"""Text-conditioned multi-operator frequency branch (spec 2.4).

Multi-band decomposition -> 16 response maps, all fp32, per-map RMS-normalized:
  - FFT:   6 radial band magnitude maps A_r                    (indices 0..5)
  - DWT:   Haar high-frequency subbands LH/HL/HH, upsampled
           back to H                                            (indices 6..8)
  - LoG:   3 scales of Laplacian-of-Gaussian, implemented as
           DoG band-pass (sigma = 1, 2, 4)                     (indices 9..11)
  - Gabor: 4 orientations (0/45/90/135 deg, lambda=4, sigma=2) (indices 12..15)

Token groups:
  - HIGH_IDX (boundary, 11 maps):        FFT[3:6] + DWT + LoG(sigma=1) + Gabor
  - MID_IDX  (texture anomaly, 4 maps):  FFT[1:3] + LoG(sigma=2, 4)

Text-controlled band selection alpha_f = Softmax(g_f(z+)) over the 16 maps is
applied by MultimodalAdapter.prepare; phi_f projects the weighted stack into
the dense spatial feature Ff [B,128,H/4,W/4]. A high-pass response A_hp
(circular 25%-area cutoff) is the edge-modality input of the fusion gate.

FFT must run fp32 (torch.fft has no bf16 support); the adapter calls
decompose()/project() inside an autocast-disabled block.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian1d(sigma):
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur2d_kernel(sigma):
    """Separable 2D Gaussian as a [1,1,k,k] kernel."""
    k1 = _gaussian1d(sigma)
    k2 = (k1[:, None] * k1[None, :])
    return k2[None, None]


def _gabor_kernel(theta_deg, lam=4.0, sigma=2.0, ksize=11):
    """Gabor cosine kernel [1,1,k,k] at orientation theta (degrees)."""
    theta = math.radians(theta_deg)
    half = ksize // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32)
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)
    k = torch.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * xr / lam)
    return k[None, None]


class FrequencyPrompt(nn.Module):
    N_MAPS = 16
    HIGH_IDX = (3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15)  # boundary group (11)
    MID_IDX = (1, 2, 10, 11)                          # texture-anomaly group (4)

    def __init__(self, img_size=1008, n_bands=6, r_freq=0.25, out_ch=128):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands
        self.r_freq = r_freq
        assert n_bands == 6  # group indices above assume the 6-ring layout

        # FFT radial band masks: equal-width rings in normalized radius [0, 1]
        N = img_size
        u = torch.arange(N, dtype=torch.float32) - N // 2
        rr = torch.sqrt(u[None, :] ** 2 + u[:, None] ** 2) / (N / 2.0)
        edges = torch.linspace(0.0, 1.0, n_bands + 1)
        masks = []
        for b in range(n_bands):
            lo = rr >= edges[b]
            hi = rr < edges[b + 1] if b < n_bands - 1 else rr <= 2.0  # last band takes the rest
            masks.append((lo & hi).float())
        self.register_buffer("band_masks", torch.stack(masks)[:, None])  # [6,1,H,W]

        # high-pass mask: circular cutoff keeping ~r_freq of the spectrum area
        # (pi * r_hp^2 = r_freq -> r_hp = sqrt(r_freq / pi) ~ 0.282)
        r_hp = math.sqrt(r_freq / math.pi)
        self.register_buffer("hp_mask", (rr >= r_hp).float()[None, None])  # [1,1,H,W]

        # DWT: fixed Haar high-frequency subband kernels (LH, HL, HH), stride 2
        haar = (torch.tensor([
            [[1.0, 1.0], [-1.0, -1.0]],   # LH (vertical detail)
            [[1.0, -1.0], [1.0, -1.0]],   # HL (horizontal detail)
            [[1.0, -1.0], [-1.0, 1.0]],   # HH (diagonal detail)
        ]) / 2.0).unsqueeze(1)  # [3,1,2,2]
        self.register_buffer("haar_w", haar)

        # LoG: DoG band-pass per scale, sigma in {1, 2, 4}
        self.log_sigmas = (1.0, 2.0, 4.0)
        for i, s in enumerate(self.log_sigmas):
            self.register_buffer(f"dog_a_{i}", _gaussian_blur2d_kernel(s))
            self.register_buffer(f"dog_b_{i}", _gaussian_blur2d_kernel(s * math.sqrt(2.0)))

        # Gabor: 4 orientations
        self.gabor_thetas = (0.0, 45.0, 90.0, 135.0)
        self.register_buffer(
            "gabor_w",
            torch.cat([_gabor_kernel(t) for t in self.gabor_thetas], dim=0),  # [4,1,11,11]
        )

        # phi_f: [B,16,H,W] -> [B,out_ch,H/4,W/4]
        self.phi_f = nn.Sequential(
            nn.Conv2d(self.N_MAPS, out_ch // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

    @staticmethod
    def _rms_norm(m, eps=1e-6):
        """Per-sample, per-map RMS normalization so all 16 maps are comparable."""
        rms = m.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        return m / (rms + eps)

    def decompose(self, gray):
        """gray: [B,1,H,W] fp32 -> (maps [B,16,H,W], hp [B,1,H,W]) fp32."""
        # FFT radial bands
        freq = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"))  # [B,1,H,W]
        band_freq = (freq[:, None] * self.band_masks[None]).squeeze(2)  # [B,6,H,W]
        fft_bands = torch.abs(
            torch.fft.ifft2(torch.fft.ifftshift(band_freq, dim=(-2, -1)), norm="ortho")
        )
        hp = torch.abs(
            torch.fft.ifft2(
                torch.fft.ifftshift(freq * self.hp_mask, dim=(-2, -1)),
                norm="ortho",
            )
        )  # [B,1,H,W]

        # Haar DWT high subbands at H/2, upsampled back to H (magnitude response)
        H, W = gray.shape[-2:]
        dwt = F.conv2d(gray, self.haar_w, stride=2).abs()  # [B,3,H/2,W/2]
        dwt = F.interpolate(dwt, size=(H, W), mode="nearest")

        # LoG (DoG band-pass): |blur(sigma*sqrt2) - blur(sigma)|
        logs = []
        for i in range(len(self.log_sigmas)):
            a = F.conv2d(gray, getattr(self, f"dog_a_{i}"), padding="same")
            b = F.conv2d(gray, getattr(self, f"dog_b_{i}"), padding="same")
            logs.append((b - a).abs())
        log_maps = torch.cat(logs, dim=1)  # [B,3,H,W]

        # Gabor energy responses (abs of cosine filter)
        gab = F.conv2d(gray, self.gabor_w, padding=5).abs()  # [B,4,H,W]

        maps = torch.cat([fft_bands, dwt, log_maps, gab], dim=1)  # [B,16,H,W]
        maps = self._rms_norm(maps)
        return maps, hp

    def project(self, weighted_maps):
        """weighted_maps: [B,16,H,W] -> Ff [B,out_ch,H/4,W/4]."""
        return self.phi_f(weighted_maps)
